//! First-party CUDA acceleration for the Goldilocks/BLAKE3 prover pipeline.
//!
//! [`CudaDft`] implements Plonky3's host-oriented [`TwoAdicSubgroupDft`]
//! contract, while the resident types in this module keep LDEs, Merkle trees,
//! lookup traces, quotient evaluations, and FRI codewords on the selected GPU.
//! All public protocol types and serialized proofs remain unchanged.

pub(crate) mod mmcs;
#[doc(hidden)]
pub mod pcs;

use core::ffi::{CStr, c_char, c_void};
use core::mem::{align_of, size_of};
use core::ptr::NonNull;
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};

use crate::expr::{RowOffset, Source};
use crate::graph::{ConstraintGraph, Node};
use p3_dft::{Radix2DitParallel, TwoAdicSubgroupDft};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64, TwoAdicField};
use p3_goldilocks::Goldilocks;
use p3_matrix::bitrev::{BitReversalPerm, BitReversedMatrixView};
use p3_matrix::dense::RowMajorMatrix;
use p3_matrix::{Dimensions, Matrix};
use p3_merkle_tree::PrunedMerklePaths;
use p3_util::log2_strict_usize;

const _: () = assert!(size_of::<Goldilocks>() == size_of::<u64>());
const _: () = assert!(align_of::<Goldilocks>() == align_of::<u64>());

type CachedPowers = Arc<[Goldilocks]>;
type SharedPowerCache<Key> = Arc<RwLock<BTreeMap<Key, CachedPowers>>>;

/// CUDA-backed batched DFT for the Goldilocks field.
///
/// Clones share the host twiddle cache. This is important because the
/// production configuration gives one clone to the PCS and retains another
/// for quotient transforms.
#[derive(Clone, Debug)]
pub struct CudaDft {
    device_id: i32,
    cpu: Radix2DitParallel<Goldilocks>,
    twiddles: SharedPowerCache<(usize, bool)>,
    shift_powers: SharedPowerCache<(usize, u64)>,
}

impl Default for CudaDft {
    fn default() -> Self {
        Self::new(configured_device())
    }
}

pub(crate) fn configured_device() -> i32 {
    std::env::var("MULTI_STARK_CUDA_DEVICE").map_or(0, |value| {
        value
            .parse()
            .expect("MULTI_STARK_CUDA_DEVICE must be a non-negative integer")
    })
}

impl CudaDft {
    /// Selects the zero-based CUDA device used by subsequent transforms.
    #[must_use]
    pub fn new(device_id: i32) -> Self {
        assert!(device_id >= 0, "CUDA device id must be non-negative");
        let _ = device_memory_info(device_id);
        Self {
            device_id,
            cpu: Radix2DitParallel::default(),
            twiddles: Arc::default(),
            shift_powers: Arc::default(),
        }
    }

    /// Returns the selected CUDA device id.
    #[must_use]
    pub const fn device_id(&self) -> i32 {
        self.device_id
    }

    fn twiddles(&self, log_height: usize, inverse: bool) -> Arc<[Goldilocks]> {
        let key = (log_height, inverse);
        if let Some(twiddles) = self
            .twiddles
            .read()
            .expect("twiddle cache poisoned")
            .get(&key)
        {
            return Arc::clone(twiddles);
        }

        let mut cache = self.twiddles.write().expect("twiddle cache poisoned");
        Arc::clone(cache.entry(key).or_insert_with(|| {
            let root = Goldilocks::two_adic_generator(log_height);
            let root = if inverse { root.inverse() } else { root };
            root.powers().take((1 << log_height) / 2).collect().into()
        }))
    }

    fn shift_powers(&self, height: usize, shift: Goldilocks) -> Arc<[Goldilocks]> {
        let key = (height, shift.as_canonical_u64());
        if let Some(powers) = self
            .shift_powers
            .read()
            .expect("shift-power cache poisoned")
            .get(&key)
        {
            return Arc::clone(powers);
        }

        let mut cache = self
            .shift_powers
            .write()
            .expect("shift-power cache poisoned");
        Arc::clone(
            cache
                .entry(key)
                .or_insert_with(|| shift.powers().take(height).collect().into()),
        )
    }

    fn validate_dimensions(height: usize, width: usize) {
        assert!(
            height.is_power_of_two(),
            "DFT height must be a power of two"
        );
        let log_height = log2_strict_usize(height);
        assert!(
            log_height <= Goldilocks::TWO_ADICITY,
            "DFT height exceeds Goldilocks two-adicity"
        );
        height
            .checked_mul(width)
            .expect("DFT matrix element count overflows usize");
    }

    #[inline]
    fn use_cuda_dft(height: usize, width: usize) -> bool {
        height.saturating_mul(width) >= (1 << 15)
    }

    #[inline]
    fn use_cuda_coset_lde(extended_height: usize, width: usize) -> bool {
        extended_height >= (1 << 15) && width <= 2
    }

    /// Reports whether `dft_batch` will execute on CUDA for this shape.
    #[must_use]
    pub fn uses_cuda_dft(height: usize, width: usize) -> bool {
        height.saturating_mul(width) >= (1 << 15)
    }

    /// Reports whether `coset_lde_batch` will execute on CUDA for this shape.
    #[must_use]
    pub fn uses_cuda_coset_lde(height: usize, width: usize, added_bits: usize) -> bool {
        let Ok(added_bits) = u32::try_from(added_bits) else {
            return false;
        };
        let Some(extended_height) = height.checked_shl(added_bits) else {
            return false;
        };
        extended_height >= (1 << 15) && width <= 2
    }

    /// Computes a coset LDE whose bit-reversed storage remains on the GPU.
    #[must_use]
    pub(crate) fn coset_lde_batch_resident(
        &self,
        matrix: &RowMajorMatrix<Goldilocks>,
        added_bits: usize,
        shift: Goldilocks,
    ) -> CudaLde {
        let height = matrix.height();
        let width = matrix.width();
        Self::validate_dimensions(height, width);
        assert!(width > 0, "resident CUDA LDE requires at least one column");
        let extended_height = height
            .checked_shl(u32::try_from(added_bits).expect("LDE blowup exceeds u32"))
            .expect("LDE height overflows usize");
        Self::validate_dimensions(extended_height, width);

        let log_height = log2_strict_usize(height);
        let inverse_twiddles = self.twiddles(log_height, true);
        let forward_twiddles = self.twiddles(log2_strict_usize(extended_height), false);
        let shift_powers = self.shift_powers(height, shift);
        let height_inverse = Goldilocks::ONE.div_2exp_u64(log_height as u64);
        let mut handle = core::ptr::null_mut();
        // SAFETY: every host buffer has the exact dimensions validated above;
        // successful creation transfers the device allocation to `CudaLde`.
        let status = unsafe {
            multi_stark_cuda_coset_lde_create(
                self.device_id,
                &mut handle,
                matrix.values.as_ptr().cast(),
                height,
                width,
                added_bits,
                inverse_twiddles.as_ptr().cast(),
                shift_powers.as_ptr().cast(),
                forward_twiddles.as_ptr().cast(),
                raw_u64(height_inverse),
            )
        };
        check_cuda(status, "resident coset LDE");
        CudaLde {
            device_id: self.device_id,
            handle: NonNull::new(handle).expect("CUDA returned a null LDE handle"),
            height: extended_height,
            width,
        }
    }

    pub(crate) fn prepare_coset_lde_constants(
        &self,
        height: usize,
        added_bits: usize,
        shift: Goldilocks,
    ) {
        let extended_height = height
            .checked_shl(u32::try_from(added_bits).expect("LDE blowup exceeds u32"))
            .expect("LDE height overflows usize");
        Self::validate_dimensions(height, 1);
        Self::validate_dimensions(extended_height, 1);
        let inverse_twiddles = self.twiddles(log2_strict_usize(height), true);
        let shift_powers = self.shift_powers(height, shift);
        let forward_twiddles = self.twiddles(log2_strict_usize(extended_height), false);
        let status = unsafe {
            multi_stark_cuda_prepare_lde_constants(
                self.device_id,
                inverse_twiddles.as_ptr().cast(),
                inverse_twiddles.len(),
                shift_powers.as_ptr().cast(),
                height,
                forward_twiddles.as_ptr().cast(),
                forward_twiddles.len(),
            )
        };
        check_cuda(status, "prepare resident LDE constants");
    }
}

/// Bit-reversed coset-LDE storage owned by a CUDA device allocation.
#[doc(hidden)]
pub struct CudaLde {
    device_id: i32,
    handle: NonNull<c_void>,
    height: usize,
    width: usize,
}

unsafe impl Send for CudaLde {}
unsafe impl Sync for CudaLde {}

impl CudaLde {
    pub(crate) const fn raw_handle(&self) -> *const c_void {
        self.handle.as_ptr()
    }

    /// # Safety
    ///
    /// The caller must serialize attachment changes with all CUDA operations
    /// that can read the attached trace.
    pub(crate) unsafe fn release_trace(&self) {
        let status =
            unsafe { multi_stark_cuda_lde_release_trace(self.device_id, self.handle.as_ptr()) };
        check_cuda(status, "resident trace release");
    }

    /// Releases the device values after they have been materialized on the
    /// host. The handle remains valid for dimensions and eventual destruction.
    ///
    /// # Safety
    ///
    /// The caller must serialize this transition with every operation which
    /// can access the LDE or its attached trace.
    pub(crate) unsafe fn release_values(&self) {
        let status =
            unsafe { multi_stark_cuda_lde_release_values(self.device_id, self.handle.as_ptr()) };
        check_cuda(status, "resident LDE value release");
    }

    /// # Safety
    ///
    /// `trace` must outlive this handle or an earlier `release_trace` call,
    /// and the caller must serialize attachment changes with CUDA trace use.
    pub(crate) unsafe fn attach_trace(&self, trace: &RowMajorMatrix<Goldilocks>) {
        assert_eq!(trace.width(), self.width);
        let status = unsafe {
            multi_stark_cuda_lde_attach_trace(
                self.device_id,
                self.handle.as_ptr(),
                trace.values.as_ptr().cast(),
                trace.height(),
                trace.width(),
            )
        };
        check_cuda(status, "resident trace attachment");
    }

    pub(crate) fn fri_fold(
        &self,
        next: Option<&CudaReducedOpening>,
        betas: &[[Goldilocks; 2]],
        beta_power: [Goldilocks; 2],
        g_inv: Goldilocks,
        ext_w: Goldilocks,
    ) -> Self {
        let mut handle = core::ptr::null_mut();
        let status = unsafe {
            multi_stark_cuda_fri_fold_resident(
                self.device_id,
                &mut handle,
                self.raw_handle(),
                next.map_or(core::ptr::null(), |value| value.handle.as_ptr()),
                betas.as_ptr().cast(),
                betas.len(),
                raw_u64(beta_power[0]),
                raw_u64(beta_power[1]),
                raw_u64(g_inv),
                raw_u64(ext_w),
            )
        };
        check_cuda(status, "resident CUDA FRI fold");
        Self {
            device_id: self.device_id,
            handle: NonNull::new(handle).expect("CUDA returned a null folded codeword"),
            height: self.height >> betas.len(),
            width: 2,
        }
    }
    /// Uploads an already bit-reversed LDE matrix into resident storage.
    #[must_use]
    pub(crate) fn from_row_major_matrix(
        device_id: i32,
        matrix: &RowMajorMatrix<Goldilocks>,
    ) -> Self {
        assert!(
            matrix.height().is_power_of_two(),
            "LDE height is not a power of two"
        );
        assert!(matrix.width() > 0, "LDE width is zero");
        let mut handle = core::ptr::null_mut();
        // SAFETY: the matrix is contiguous and remains live for the synchronous upload.
        let status = unsafe {
            multi_stark_cuda_lde_create_from_host(
                device_id,
                &mut handle,
                matrix.values.as_ptr().cast(),
                matrix.height(),
                matrix.width(),
            )
        };
        check_cuda(status, "resident LDE upload");
        Self {
            device_id,
            handle: NonNull::new(handle).expect("CUDA returned a null LDE handle"),
            height: matrix.height(),
            width: matrix.width(),
        }
    }

    #[must_use]
    pub(crate) const fn height(&self) -> usize {
        self.height
    }

    #[must_use]
    pub(crate) const fn width(&self) -> usize {
        self.width
    }

    /// Copies the bit-reversed committed storage to a host matrix. This is an
    /// oracle/debug escape hatch; resident PCS code should retain the handle.
    #[must_use]
    pub(crate) fn to_row_major_matrix(&self) -> RowMajorMatrix<Goldilocks> {
        let len = self.height * self.width;
        let mut storage = Vec::<core::mem::MaybeUninit<Goldilocks>>::with_capacity(len);
        // SAFETY: the CUDA copy below initializes every byte before the storage
        // is converted to `Vec<Goldilocks>`. On failure it remains a safely
        // droppable vector of `MaybeUninit` values.
        unsafe { storage.set_len(len) };
        // SAFETY: the output allocation matches the immutable resident LDE.
        let status = unsafe {
            multi_stark_cuda_lde_copy_to_host(
                self.device_id,
                self.handle.as_ptr(),
                storage.as_mut_ptr().cast(),
            )
        };
        check_cuda(status, "resident LDE host copy");
        let mut storage = core::mem::ManuallyDrop::new(storage);
        // SAFETY: the successful FFI call initialized all elements, Goldilocks
        // has the same layout inside `MaybeUninit`, and ownership moves exactly
        // once into the returned vector.
        let values = unsafe {
            Vec::from_raw_parts(
                storage.as_mut_ptr().cast::<Goldilocks>(),
                storage.len(),
                storage.capacity(),
            )
        };
        RowMajorMatrix::new(values, self.width)
    }

    /// Copies one sampled row from the resident bit-reversed LDE.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn row(&self, row: usize) -> Vec<Goldilocks> {
        assert!(row < self.height, "resident LDE row index out of bounds");
        let mut values = Goldilocks::zero_vec(self.width);
        // SAFETY: `row` is in bounds and the output has exactly `width`
        // elements, matching the contiguous device row.
        let status = unsafe {
            multi_stark_cuda_lde_copy_row(
                self.device_id,
                self.handle.as_ptr(),
                row,
                values.as_mut_ptr().cast(),
            )
        };
        check_cuda(status, "resident LDE row copy");
        values
    }

    /// Gathers sampled rows with one device kernel and one host transfer.
    #[must_use]
    pub(crate) fn rows(&self, rows: &[usize]) -> Vec<Vec<Goldilocks>> {
        assert!(!rows.is_empty(), "resident LDE row batch is empty");
        assert!(
            rows.iter().all(|&row| row < self.height),
            "resident LDE row index out of bounds"
        );
        let device_rows: Vec<u64> = rows
            .iter()
            .map(|&row| u64::try_from(row).expect("row index exceeds u64"))
            .collect();
        let mut values = Goldilocks::zero_vec(rows.len() * self.width);
        // SAFETY: all row indices are in bounds and `values` holds one full
        // output row for every requested index.
        let status = unsafe {
            multi_stark_cuda_lde_copy_rows(
                self.device_id,
                self.handle.as_ptr(),
                device_rows.as_ptr(),
                device_rows.len(),
                values.as_mut_ptr().cast(),
            )
        };
        check_cuda(status, "resident LDE row batch copy");
        values
            .chunks_exact(self.width)
            .map(<[Goldilocks]>::to_vec)
            .collect()
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaConstraintNode {
    value: u64,
    a: u32,
    b: u32,
    op: u32,
    aux: u32,
    out: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct CudaConstraintLookup {
    multiplicity: u32,
    arg_start: u32,
    arg_count: u32,
    emit_after: u32,
    output: u32,
}

type EncodedLookupGraph = (
    Vec<CudaConstraintNode>,
    usize,
    Vec<CudaConstraintLookup>,
    Vec<u32>,
);

fn checked_u32(value: usize, what: &str) -> u32 {
    u32::try_from(value).unwrap_or_else(|_| panic!("{what} exceeds the CUDA ABI limit"))
}

fn encode_constraint_nodes(graph: &ConstraintGraph<Goldilocks>) -> Vec<CudaConstraintNode> {
    graph
        .nodes
        .iter()
        .enumerate()
        .map(|(index, node)| {
            let mut out = CudaConstraintNode {
                value: 0,
                a: u32::MAX,
                b: u32::MAX,
                op: 0,
                aux: 0,
                out: checked_u32(index, "constraint node index"),
            };
            match *node {
                Node::Const(value) => out.value = raw_u64(value),
                Node::Var(col) => {
                    out.op = 1;
                    out.a = col.index;
                    out.aux = match col.source {
                        Source::Preprocessed => 0,
                        Source::Main => 2,
                        Source::Stage2 => 4,
                    } + u32::from(matches!(col.offset, RowOffset::Next));
                }
                Node::Public(i) => {
                    out.op = 2;
                    out.a = i
                }
                Node::IsFirstRow => out.op = 3,
                Node::IsLastRow => out.op = 4,
                Node::IsTransition => out.op = 5,
                Node::Add(a, b) => {
                    out.op = 6;
                    out.a = a.0;
                    out.b = b.0
                }
                Node::Sub(a, b) => {
                    out.op = 7;
                    out.a = a.0;
                    out.b = b.0
                }
                Node::Mul(a, b) => {
                    out.op = 8;
                    out.a = a.0;
                    out.b = b.0
                }
                Node::Neg(a) => {
                    out.op = 9;
                    out.a = a.0
                }
            }
            out
        })
        .collect()
}

fn encode_quotient_nodes(
    graph: &ConstraintGraph<Goldilocks>,
) -> (Vec<CudaConstraintNode>, Vec<u32>, usize) {
    let n = graph.nodes.len();
    let mut last: Vec<_> = (0..n).collect();
    for (i, node) in graph.nodes.iter().enumerate() {
        match *node {
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                last[a.index()] = i;
                last[b.index()] = i
            }
            Node::Neg(a) => last[a.index()] = i,
            _ => {}
        }
    }
    for id in graph.zeros.iter().chain(
        graph
            .lookups
            .iter()
            .flat_map(|lookup| core::iter::once(&lookup.multiplicity).chain(lookup.args.iter())),
    ) {
        last[id.index()] = n;
    }
    let mut slots = vec![0u32; n];
    let mut free = Vec::new();
    let mut retire = vec![Vec::new(); n + 1];
    let mut count = 0u32;
    for i in 0..n {
        free.append(&mut retire[i]);
        let slot = free.pop().unwrap_or_else(|| {
            let s = count;
            count += 1;
            s
        });
        slots[i] = slot;
        if last[i] < n {
            retire[last[i] + 1].push(slot);
        }
    }
    let mut nodes = encode_constraint_nodes(graph);
    for (i, (encoded, node)) in nodes.iter_mut().zip(&graph.nodes).enumerate() {
        encoded.out = slots[i];
        match *node {
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                encoded.a = slots[a.index()];
                encoded.b = slots[b.index()]
            }
            Node::Neg(a) => encoded.a = slots[a.index()],
            _ => {}
        }
    }
    (nodes, slots, count as usize)
}

/// Conservative device-memory requirement for one fused quotient job.
///
/// The graph evaluator reuses slots as soon as their final consumer has run,
/// so `graph.nodes.len()` can be orders of magnitude larger than the live
/// device scratch. Keep this estimate beside the encoder so admission and the
/// kernel use the same liveness calculation. The scratch term assumes the
/// global-memory path; devices able to fit the slots in shared memory need
/// less than this bound.
pub(crate) fn quotient_lde_memory_upper_bound(
    graph: &ConstraintGraph<Goldilocks>,
    public_count: usize,
    constraint_count: usize,
    quotient_size: usize,
    quotient_degree: usize,
    log_blowup: usize,
) -> (usize, usize) {
    let (nodes, _, slot_count) = encode_quotient_nodes(graph);
    let lookup_arg_count = graph
        .lookups
        .iter()
        .map(|lookup| lookup.args.len())
        .sum::<usize>();
    let align8 = |bytes: usize| bytes.saturating_add(7) & !7;
    let allocation_bytes = [
        nodes.len().saturating_mul(size_of::<CudaConstraintNode>()),
        graph.zeros.len().saturating_mul(size_of::<u32>()),
        graph
            .lookups
            .len()
            .saturating_mul(size_of::<CudaConstraintLookup>()),
        lookup_arg_count.saturating_mul(size_of::<u32>()),
        public_count.saturating_mul(size_of::<Goldilocks>()),
        4usize
            .saturating_mul(quotient_size)
            .saturating_mul(size_of::<Goldilocks>()),
        2usize
            .saturating_mul(constraint_count)
            .saturating_mul(size_of::<Goldilocks>()),
        2 * size_of::<Goldilocks>(),
        2usize
            .saturating_mul(quotient_size)
            .saturating_mul(size_of::<Goldilocks>()),
    ]
    .into_iter()
    .map(align8)
    .sum::<usize>();
    let global_blocks = quotient_size.div_ceil(128).min(256);
    let scratch_bytes = global_blocks
        .saturating_mul(slot_count)
        .saturating_mul(128)
        .saturating_mul(size_of::<Goldilocks>());
    let trace_height = quotient_size / quotient_degree;
    let lde_height = trace_height
        .checked_shl(u32::try_from(log_blowup).expect("LDE blowup exceeds u32"))
        .unwrap_or(usize::MAX);
    let output_bytes = lde_height
        .saturating_mul(2)
        .saturating_mul(quotient_degree)
        .saturating_mul(size_of::<Goldilocks>());
    (output_bytes, allocation_bytes.saturating_add(scratch_bytes))
}

fn encode_lookup_nodes(graph: &ConstraintGraph<Goldilocks>) -> Option<EncodedLookupGraph> {
    let n = graph.lookup_prefix_len;
    if n == 0 || graph.lookups.is_empty() {
        return None;
    }
    let mut reachable = vec![false; n];
    let mut pending: Vec<usize> = graph
        .lookups
        .iter()
        .flat_map(|lookup| {
            core::iter::once(lookup.multiplicity.index())
                .chain(lookup.args.iter().map(|id| id.index()))
        })
        .collect();
    while let Some(index) = pending.pop() {
        if index >= n || reachable[index] {
            continue;
        }
        reachable[index] = true;
        match graph.nodes[index] {
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                pending.push(a.index());
                pending.push(b.index());
            }
            Node::Neg(a) => pending.push(a.index()),
            _ => {}
        }
    }
    let order: Vec<_> = reachable
        .iter()
        .enumerate()
        .filter_map(|(index, &used)| used.then_some(index))
        .collect();
    let mut compact = vec![usize::MAX; n];
    for (position, &index) in order.iter().enumerate() {
        compact[index] = position;
    }
    let node_count = order.len();
    // Lookup witness expressions are base-trace expressions. Public values
    // are the logUp challenges sampled only after stage 1, and stage 2 does
    // not exist yet, so neither can legally occur in this prefix.
    if order.iter().map(|&index| &graph.nodes[index]).any(|node| {
        matches!(node, Node::Public(_))
            || matches!(node, Node::Var(col) if col.source == Source::Stage2)
    }) {
        return None;
    }
    let mut last = vec![0usize; node_count];
    for (position, &index) in order.iter().enumerate() {
        let node = &graph.nodes[index];
        match *node {
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                last[compact[a.index()]] = position;
                last[compact[b.index()]] = position;
            }
            Node::Neg(a) => last[compact[a.index()]] = position,
            _ => {}
        }
    }
    let emit_after = graph
        .lookups
        .iter()
        .map(|lookup| {
            core::iter::once(&lookup.multiplicity)
                .chain(lookup.args.iter())
                .map(|id| compact[id.index()])
                .max()
                .expect("lookup has a multiplicity")
        })
        .collect::<Vec<_>>();
    for (lookup, &emit) in graph.lookups.iter().zip(&emit_after) {
        for id in core::iter::once(&lookup.multiplicity).chain(lookup.args.iter()) {
            last[compact[id.index()]] = last[compact[id.index()]].max(emit);
        }
    }
    let mut slots = vec![0u32; node_count];
    let mut free = Vec::new();
    let mut retire = vec![Vec::new(); node_count + 1];
    let mut slot_count = 0u32;
    for i in 0..node_count {
        free.append(&mut retire[i]);
        let slot = free.pop().unwrap_or_else(|| {
            let result = slot_count;
            slot_count += 1;
            result
        });
        slots[i] = slot;
        if last[i] < node_count {
            retire[last[i] + 1].push(slot);
        }
    }
    let encoded_all = encode_constraint_nodes(graph);
    let mut nodes: Vec<_> = order.iter().map(|&index| encoded_all[index]).collect();
    for (position, (&index, encoded)) in order.iter().zip(&mut nodes).enumerate() {
        let node = &graph.nodes[index];
        encoded.out = slots[position];
        match *node {
            Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) => {
                encoded.a = slots[compact[a.index()]];
                encoded.b = slots[compact[b.index()]];
            }
            Node::Neg(a) => encoded.a = slots[compact[a.index()]],
            _ => {}
        }
    }
    let mut args = Vec::new();
    let mut lookups = graph
        .lookups
        .iter()
        .zip(emit_after)
        .enumerate()
        .map(|(output, (lookup, emit_after))| {
            let arg_start = checked_u32(args.len(), "lookup argument offset");
            args.extend(lookup.args.iter().map(|id| slots[compact[id.index()]]));
            CudaConstraintLookup {
                multiplicity: slots[compact[lookup.multiplicity.index()]],
                arg_start,
                arg_count: checked_u32(lookup.args.len(), "lookup argument count"),
                emit_after: checked_u32(emit_after, "lookup emission index"),
                output: checked_u32(output, "lookup output index"),
            }
        })
        .collect::<Vec<_>>();
    lookups.sort_unstable_by_key(|lookup| lookup.emit_after);
    Some((nodes, slot_count as usize, lookups, args))
}

/// Conservative device-memory requirement for one graph-based lookup job.
///
/// Keep this calculation beside [`encode_lookup_nodes`]: the CUDA entry point
/// evaluates lookup expressions directly from the retained trace in bounded
/// row chunks. In particular, it never materializes the full concrete lookup
/// argument matrix used by the host fallback.
pub(crate) fn lookup_graph_lde_memory_upper_bound(
    graph: &ConstraintGraph<Goldilocks>,
    height: usize,
    main_width: usize,
    group_size: usize,
    log_blowup: usize,
) -> Option<(usize, usize)> {
    let (nodes, slot_count, lookups, args) = encode_lookup_nodes(graph)?;
    let lookup_count = lookups.len();
    let groups = lookup_count.div_ceil(group_size.max(1));
    let extended_height = height
        .checked_shl(u32::try_from(log_blowup).expect("LDE blowup exceeds u32"))
        .unwrap_or(usize::MAX);
    let output_bytes = extended_height
        .saturating_mul(groups)
        .saturating_mul(2 * size_of::<Goldilocks>());

    const LOOKUP_ROWS_PER_CHUNK: usize = 1 << 16;
    let chunk_rows = height.min(LOOKUP_ROWS_PER_CHUNK);
    let message_count = chunk_rows.saturating_mul(lookup_count);
    let align8 = |bytes: usize| bytes.saturating_add(7) & !7;
    let metadata_bytes = [
        nodes.len().saturating_mul(size_of::<CudaConstraintNode>()),
        lookups
            .len()
            .saturating_mul(size_of::<CudaConstraintLookup>()),
        args.len().saturating_mul(size_of::<u32>()),
    ]
    .into_iter()
    .map(align8)
    .sum::<usize>();
    // Per message: conjugate Ext2, norm, inverse norm, multiplicity.
    let message_bytes = message_count.saturating_mul(5 * size_of::<Goldilocks>());
    let delta_bytes = height
        .saturating_mul(groups)
        .saturating_mul(2 * size_of::<Goldilocks>());
    let shared_tile = (48 * 1024) / slot_count.saturating_mul(size_of::<Goldilocks>()).max(1);
    let scratch_bytes = if shared_tile < 32 {
        chunk_rows
            .div_ceil(128)
            .min(256)
            .saturating_mul(slot_count)
            .saturating_mul(128 * size_of::<Goldilocks>())
    } else {
        0
    };
    // Some stage-1 policies retain the original trace in pinned host memory.
    // Charge one bounded staging chunk even when this particular LDE still
    // owns a device trace, keeping admission safe across both representations.
    let trace_chunk_bytes = chunk_rows
        .saturating_add(1)
        .saturating_mul(main_width)
        .saturating_mul(size_of::<Goldilocks>());
    // Device-cached twiddles and shift powers may be cold for this height.
    let constant_bytes = height
        .saturating_div(2)
        .saturating_add(height)
        .saturating_add(extended_height / 2)
        .saturating_mul(size_of::<Goldilocks>());
    Some((
        output_bytes,
        metadata_bytes
            .saturating_add(message_bytes)
            .saturating_add(delta_bytes)
            .saturating_add(scratch_bytes)
            .saturating_add(trace_chunk_bytes)
            .saturating_add(constant_bytes),
    ))
}

/// Evaluates the compiled base-field constraint roots directly against
/// resident trace LDEs. This is the protocol-independent core used by the
/// CUDA quotient path; selectors and public values remain caller supplied.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn constraint_graph_roots(
    graph: &ConstraintGraph<Goldilocks>,
    preprocessed: Option<&CudaLde>,
    main: &CudaLde,
    stage2: &CudaLde,
    publics: &[Goldilocks],
    selectors: &[Goldilocks],
    quotient_size: usize,
    next_step: usize,
) -> RowMajorMatrix<Goldilocks> {
    assert_eq!(selectors.len(), 3 * quotient_size);
    assert!(next_step <= quotient_size);
    let nodes = encode_constraint_nodes(graph);
    let roots: Vec<u32> = graph.zeros.iter().map(|root| root.0).collect();
    let mut output = Goldilocks::zero_vec(quotient_size * roots.len());
    let status = unsafe {
        multi_stark_cuda_constraint_graph(
            main.device_id,
            output.as_mut_ptr().cast(),
            nodes.as_ptr().cast(),
            nodes.len(),
            roots.as_ptr().cast(),
            roots.len(),
            preprocessed.map_or(core::ptr::null(), CudaLde::raw_handle),
            main.raw_handle(),
            stage2.raw_handle(),
            publics.as_ptr().cast(),
            publics.len(),
            selectors.as_ptr().cast(),
            quotient_size,
            next_step,
        )
    };
    check_cuda(status, "constraint graph evaluation");
    RowMajorMatrix::new(output, roots.len())
}

/// Compact description of the four Lagrange-selector columns used on a
/// two-adic quotient coset. CUDA expands these geometric sequences directly
/// into device memory instead of receiving four full host vectors.
#[derive(Clone, Copy)]
pub(crate) struct CudaCosetSelectors {
    pub(crate) coset_shift: Goldilocks,
    pub(crate) coset_generator: Goldilocks,
    pub(crate) trace_last: Goldilocks,
    pub(crate) vanishing_start: Goldilocks,
    pub(crate) vanishing_step: Goldilocks,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn quotient_values_resident(
    graph: &ConstraintGraph<Goldilocks>,
    preprocessed: Option<&CudaLde>,
    main: &CudaLde,
    stage2: &CudaLde,
    publics: &[Goldilocks],
    selectors: CudaCosetSelectors,
    alpha: &[Goldilocks],
    delta: &[Goldilocks; 2],
    ext_w: Goldilocks,
    quotient_size: usize,
    next_step: usize,
    group_size: usize,
) -> RowMajorMatrix<Goldilocks> {
    assert!((1..=8).contains(&group_size));
    assert!(publics.len() >= 4);
    assert!(next_step.is_power_of_two());
    assert!(next_step <= quotient_size);
    let (nodes, slots, slot_count) = encode_quotient_nodes(graph);
    let roots: Vec<_> = graph.zeros.iter().map(|root| slots[root.index()]).collect();
    let mut args = Vec::new();
    let lookups: Vec<_> = graph
        .lookups
        .iter()
        .map(|lookup| {
            let arg_start = checked_u32(args.len(), "lookup argument offset");
            args.extend(lookup.args.iter().map(|arg| slots[arg.index()]));
            CudaConstraintLookup {
                multiplicity: slots[lookup.multiplicity.index()],
                arg_start,
                arg_count: checked_u32(lookup.args.len(), "lookup argument count"),
                emit_after: 0,
                output: 0,
            }
        })
        .collect();
    let constraint_count = alpha.len() / 2;
    let expected_constraints = roots.len()
        + if lookups.is_empty() {
            2
        } else {
            2 * lookups.len().div_ceil(group_size)
        };
    assert_eq!(alpha.len(), 2 * expected_constraints);
    let mut output = Goldilocks::zero_vec(2 * quotient_size);
    let status = unsafe {
        multi_stark_cuda_quotient_values(
            main.device_id,
            output.as_mut_ptr().cast(),
            nodes.as_ptr().cast(),
            nodes.len(),
            slot_count,
            roots.as_ptr(),
            roots.len(),
            lookups.as_ptr().cast(),
            lookups.len(),
            args.as_ptr(),
            args.len(),
            group_size,
            preprocessed.map_or(core::ptr::null(), CudaLde::raw_handle),
            main.raw_handle(),
            stage2.raw_handle(),
            publics.as_ptr().cast(),
            publics.len(),
            raw_u64(selectors.coset_shift),
            raw_u64(selectors.coset_generator),
            raw_u64(selectors.trace_last),
            raw_u64(selectors.vanishing_start),
            raw_u64(selectors.vanishing_step),
            alpha.as_ptr().cast(),
            constraint_count,
            delta.as_ptr().cast(),
            raw_u64(ext_w),
            quotient_size,
            next_step,
        )
    };
    check_cuda(status, "resident quotient evaluation");
    RowMajorMatrix::new(output, 2)
}

/// Evaluate a quotient and carry it through coefficient slicing and the
/// committed low-degree extension without materializing an intermediate host
/// matrix.  The returned storage has exactly the bit-reversed layout expected
/// by `Pcs::commit_ldes`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn quotient_lde_mixed(
    dft: &CudaDft,
    graph: &ConstraintGraph<Goldilocks>,
    preprocessed: Option<mmcs::CudaMatrixSource<'_>>,
    main: mmcs::CudaMatrixSource<'_>,
    stage2: mmcs::CudaMatrixSource<'_>,
    publics: &[Goldilocks],
    selectors: CudaCosetSelectors,
    alpha: &[Goldilocks],
    delta: &[Goldilocks; 2],
    ext_w: Goldilocks,
    quotient_size: usize,
    next_step: usize,
    group_size: usize,
    quotient_degree: usize,
    log_blowup: usize,
) -> CudaLde {
    quotient_lde_sources(
        dft,
        graph,
        preprocessed,
        main,
        stage2,
        publics,
        selectors,
        alpha,
        delta,
        ext_w,
        quotient_size,
        next_step,
        group_size,
        quotient_degree,
        log_blowup,
    )
}

#[allow(clippy::too_many_arguments)]
fn quotient_lde_sources(
    dft: &CudaDft,
    graph: &ConstraintGraph<Goldilocks>,
    preprocessed: Option<mmcs::CudaMatrixSource<'_>>,
    main: mmcs::CudaMatrixSource<'_>,
    stage2: mmcs::CudaMatrixSource<'_>,
    publics: &[Goldilocks],
    selectors: CudaCosetSelectors,
    alpha: &[Goldilocks],
    delta: &[Goldilocks; 2],
    ext_w: Goldilocks,
    quotient_size: usize,
    next_step: usize,
    group_size: usize,
    quotient_degree: usize,
    log_blowup: usize,
) -> CudaLde {
    assert!((1..=8).contains(&group_size));
    assert!(publics.len() >= 4);
    assert!(next_step.is_power_of_two());
    assert!(next_step <= quotient_size);
    assert!(quotient_degree.is_power_of_two());
    assert_eq!(quotient_size % quotient_degree, 0);
    let (nodes, slots, slot_count) = encode_quotient_nodes(graph);
    let roots: Vec<_> = graph.zeros.iter().map(|root| slots[root.index()]).collect();
    let mut args = Vec::new();
    let lookups: Vec<_> = graph
        .lookups
        .iter()
        .map(|lookup| {
            let arg_start = checked_u32(args.len(), "lookup argument offset");
            args.extend(lookup.args.iter().map(|arg| slots[arg.index()]));
            CudaConstraintLookup {
                multiplicity: slots[lookup.multiplicity.index()],
                arg_start,
                arg_count: checked_u32(lookup.args.len(), "lookup argument count"),
                emit_after: 0,
                output: 0,
            }
        })
        .collect();
    let constraint_count = alpha.len() / 2;
    let expected_constraints = roots.len()
        + if lookups.is_empty() {
            2
        } else {
            2 * lookups.len().div_ceil(group_size)
        };
    assert_eq!(alpha.len(), 2 * expected_constraints);
    let trace_height = quotient_size / quotient_degree;
    let lde_height = trace_height << log_blowup;
    let quotient_twiddles = dft.twiddles(log2_strict_usize(quotient_size), false);
    let lde_twiddles = dft.twiddles(log2_strict_usize(lde_height), false);
    let height_inverse = Goldilocks::ONE.div_2exp_u64(log2_strict_usize(quotient_size) as u64);
    let weight_step = Goldilocks::GENERATOR.exp_u64(trace_height as u64).inverse();
    let weights: Vec<_> = weight_step
        .powers()
        .take(quotient_degree)
        .map(|weight| weight * height_inverse)
        .collect();
    let source_parts =
        |source: Option<mmcs::CudaMatrixSource<'_>>| -> (*const c_void, *const u64, usize, usize) {
            match source {
                Some(mmcs::CudaMatrixSource::Resident(lde)) => (
                    lde.raw_handle(),
                    core::ptr::null(),
                    lde.height(),
                    lde.width(),
                ),
                Some(mmcs::CudaMatrixSource::Host(matrix)) => (
                    core::ptr::null(),
                    matrix.values.as_ptr().cast(),
                    matrix.height(),
                    matrix.width(),
                ),
                None => (core::ptr::null(), core::ptr::null(), 0, 0),
            }
        };
    let (preprocessed_handle, preprocessed_host, preprocessed_height, preprocessed_width) =
        source_parts(preprocessed);
    let (main_handle, main_host, main_height, main_width) = source_parts(Some(main));
    let (stage2_handle, stage2_host, stage2_height, stage2_width) = source_parts(Some(stage2));
    let mixed = !preprocessed_host.is_null() || !main_host.is_null() || !stage2_host.is_null();
    let mut handle = core::ptr::null_mut();
    let status = if mixed {
        // SAFETY: every host matrix and resident handle remains borrowed for
        // this synchronous call. The returned LDE owns its device storage.
        unsafe {
            multi_stark_cuda_quotient_lde_mixed(
                dft.device_id,
                &mut handle,
                nodes.as_ptr().cast(),
                nodes.len(),
                slot_count,
                roots.as_ptr(),
                roots.len(),
                lookups.as_ptr().cast(),
                lookups.len(),
                args.as_ptr(),
                args.len(),
                group_size,
                preprocessed_handle,
                preprocessed_host,
                preprocessed_height,
                preprocessed_width,
                main_handle,
                main_host,
                main_height,
                main_width,
                stage2_handle,
                stage2_host,
                stage2_height,
                stage2_width,
                publics.as_ptr().cast(),
                publics.len(),
                raw_u64(selectors.coset_shift),
                raw_u64(selectors.coset_generator),
                raw_u64(selectors.trace_last),
                raw_u64(selectors.vanishing_start),
                raw_u64(selectors.vanishing_step),
                alpha.as_ptr().cast(),
                constraint_count,
                delta.as_ptr().cast(),
                raw_u64(ext_w),
                quotient_size,
                next_step,
                quotient_degree,
                log_blowup,
                quotient_twiddles.as_ptr().cast(),
                lde_twiddles.as_ptr().cast(),
                weights.as_ptr().cast(),
            )
        }
    } else {
        // SAFETY: all resident handles and input slices remain live for this
        // synchronous call. The returned LDE owns its device storage.
        unsafe {
            multi_stark_cuda_quotient_lde(
                dft.device_id,
                &mut handle,
                nodes.as_ptr().cast(),
                nodes.len(),
                slot_count,
                roots.as_ptr(),
                roots.len(),
                lookups.as_ptr().cast(),
                lookups.len(),
                args.as_ptr(),
                args.len(),
                group_size,
                preprocessed_handle,
                main_handle,
                stage2_handle,
                publics.as_ptr().cast(),
                publics.len(),
                raw_u64(selectors.coset_shift),
                raw_u64(selectors.coset_generator),
                raw_u64(selectors.trace_last),
                raw_u64(selectors.vanishing_start),
                raw_u64(selectors.vanishing_step),
                alpha.as_ptr().cast(),
                constraint_count,
                delta.as_ptr().cast(),
                raw_u64(ext_w),
                quotient_size,
                next_step,
                quotient_degree,
                log_blowup,
                quotient_twiddles.as_ptr().cast(),
                lde_twiddles.as_ptr().cast(),
                weights.as_ptr().cast(),
            )
        }
    };
    check_cuda(status, "CUDA quotient LDE");
    CudaLde {
        device_id: dft.device_id,
        handle: NonNull::new(handle).expect("CUDA returned a null quotient LDE"),
        height: lde_height,
        width: 2 * quotient_degree,
    }
}

pub(crate) fn mixed_lde_open_row(ldes: &[CudaLde], index: usize) -> Vec<Vec<Goldilocks>> {
    assert!(!ldes.is_empty());
    let max_height = ldes.iter().map(CudaLde::height).max().unwrap();
    mixed_lde_open_row_at_height(ldes.iter(), max_height, index)
}

pub(crate) fn mixed_lde_open_row_at_height<'a>(
    ldes: impl IntoIterator<Item = &'a CudaLde>,
    max_height: usize,
    index: usize,
) -> Vec<Vec<Goldilocks>> {
    let ldes = ldes.into_iter().collect::<Vec<_>>();
    assert!(!ldes.is_empty());
    assert!(max_height.is_power_of_two());
    assert!(ldes.iter().all(|lde| lde.height() <= max_height));
    assert!(index < max_height);
    let total: usize = ldes.iter().map(|lde| lde.width()).sum();
    let handles: Vec<_> = ldes.iter().map(|lde| lde.raw_handle()).collect();
    let mut flat = Goldilocks::zero_vec(total);
    let status = unsafe {
        multi_stark_cuda_mixed_lde_open_row(
            ldes[0].device_id,
            flat.as_mut_ptr().cast(),
            handles.as_ptr(),
            handles.len(),
            max_height,
            index,
        )
    };
    check_cuda(status, "resident mixed LDE row opening");
    let mut offset = 0;
    ldes.iter()
        .map(|lde| {
            let row = flat[offset..offset + lde.width()].to_vec();
            offset += lde.width();
            row
        })
        .collect()
}

pub(crate) fn mixed_lde_open_rows(
    ldes: &[CudaLde],
    indices: &[usize],
) -> Vec<Vec<Vec<Goldilocks>>> {
    assert!(!ldes.is_empty());
    assert!(!indices.is_empty());
    let max_height = ldes.iter().map(CudaLde::height).max().unwrap();
    mixed_lde_open_rows_at_height(ldes.iter(), max_height, indices)
}

pub(crate) fn mixed_lde_open_rows_at_height<'a>(
    ldes: impl IntoIterator<Item = &'a CudaLde>,
    max_height: usize,
    indices: &[usize],
) -> Vec<Vec<Vec<Goldilocks>>> {
    let ldes = ldes.into_iter().collect::<Vec<_>>();
    assert!(!ldes.is_empty());
    assert!(max_height.is_power_of_two());
    assert!(ldes.iter().all(|lde| lde.height() <= max_height));
    assert!(indices.iter().all(|&index| index < max_height));
    let total: usize = ldes.iter().map(|lde| lde.width()).sum();
    let handles: Vec<_> = ldes.iter().map(|lde| lde.raw_handle()).collect();
    let device_indices: Vec<u64> = indices
        .iter()
        .map(|&index| u64::try_from(index).expect("row index exceeds u64"))
        .collect();
    let mut flat = Goldilocks::zero_vec(indices.len() * total);
    let status = unsafe {
        multi_stark_cuda_mixed_lde_open_rows(
            ldes[0].device_id,
            flat.as_mut_ptr().cast(),
            handles.as_ptr(),
            handles.len(),
            max_height,
            device_indices.as_ptr(),
            device_indices.len(),
        )
    };
    check_cuda(status, "resident mixed LDE row batch opening");
    flat.chunks_exact(total)
        .map(|query| {
            let mut offset = 0;
            ldes.iter()
                .map(|lde| {
                    let row = query[offset..offset + lde.width()].to_vec();
                    offset += lde.width();
                    row
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
pub(crate) fn lde_interpolate_ext2(
    lde: &CudaLde,
    height: usize,
    inv_denoms: &[[Goldilocks; 2]],
    coset: &[Goldilocks],
    scale: [Goldilocks; 2],
    ext_w: Goldilocks,
) -> Vec<[Goldilocks; 2]> {
    assert_eq!(inv_denoms.len(), height);
    assert_eq!(coset.len(), height);
    let mut output = vec![[Goldilocks::ZERO; 2]; lde.width()];
    let one = [Goldilocks::ONE, Goldilocks::ZERO];
    let status = unsafe {
        multi_stark_cuda_lde_interpolate(
            lde.device_id,
            output.as_mut_ptr().cast(),
            lde.raw_handle(),
            height,
            inv_denoms.as_ptr().cast(),
            coset.as_ptr().cast(),
            one.as_ptr().cast(),
            raw_u64(ext_w),
        )
    };
    check_cuda(status, "resident LDE interpolation");
    for value in &mut output {
        let [a0, a1] = *value;
        *value = [
            a0 * scale[0] + ext_w * (a1 * scale[1]),
            a0 * scale[1] + a1 * scale[0],
        ];
    }
    output
}

pub(crate) struct CudaReducedOpening {
    device_id: i32,
    handle: NonNull<c_void>,
    height: usize,
}
impl CudaReducedOpening {
    pub(crate) fn new(device_id: i32, height: usize) -> Self {
        let mut handle = core::ptr::null_mut();
        let status = unsafe { multi_stark_cuda_reduced_create(device_id, &mut handle, height) };
        check_cuda(status, "reduced opening allocation");
        Self {
            device_id,
            handle: NonNull::new(handle).unwrap(),
            height,
        }
    }
    pub(crate) const fn height(&self) -> usize {
        self.height
    }

    pub(crate) fn add_host(&mut self, values: &[[Goldilocks; 2]]) {
        assert_eq!(values.len(), self.height);
        let status = unsafe {
            multi_stark_cuda_reduced_add_host(
                self.device_id,
                self.handle.as_ptr(),
                values.as_ptr().cast(),
                values.len(),
            )
        };
        check_cuda(status, "host reduced-opening accumulation")
    }
    #[cfg(test)]
    pub(crate) fn add(
        &mut self,
        lde: &CudaLde,
        inv: &[[Goldilocks; 2]],
        alpha: &[[Goldilocks; 2]],
        reduced_y: [Goldilocks; 2],
        offset: [Goldilocks; 2],
        ext_w: Goldilocks,
    ) {
        assert_eq!(inv.len(), self.height);
        assert_eq!(alpha.len(), lde.width());
        let status = unsafe {
            multi_stark_cuda_reduced_add(
                self.device_id,
                self.handle.as_ptr(),
                lde.raw_handle(),
                self.height,
                inv.as_ptr().cast(),
                alpha.as_ptr().cast(),
                reduced_y.as_ptr().cast(),
                offset.as_ptr().cast(),
                raw_u64(ext_w),
            )
        };
        check_cuda(status, "reduced opening accumulation")
    }
    #[cfg(test)]
    pub(crate) fn to_host(&self) -> Vec<[Goldilocks; 2]> {
        let mut out = vec![[Goldilocks::ZERO; 2]; self.height];
        let status = unsafe {
            multi_stark_cuda_reduced_copy(
                self.device_id,
                self.handle.as_ptr(),
                out.as_mut_ptr().cast(),
            )
        };
        check_cuda(status, "reduced opening copy");
        out
    }
    pub(crate) fn into_lde(self) -> CudaLde {
        let mut handle = core::ptr::null_mut();
        let status = unsafe {
            multi_stark_cuda_reduced_into_lde(self.device_id, &mut handle, self.handle.as_ptr())
        };
        check_cuda(status, "reduced opening to resident FRI codeword");
        CudaLde {
            device_id: self.device_id,
            handle: NonNull::new(handle).unwrap(),
            height: self.height,
            width: 2,
        }
    }
}
impl Drop for CudaReducedOpening {
    fn drop(&mut self) {
        unsafe {
            let _ = multi_stark_cuda_reduced_destroy(self.device_id, self.handle.as_ptr());
        }
    }
}

#[repr(C)]
pub(crate) struct CudaInterpolationTask {
    pub(crate) lde: *const c_void,
    pub(crate) height: usize,
    pub(crate) inv_offset: usize,
    pub(crate) output_offset: usize,
    pub(crate) scale0: u64,
    pub(crate) scale1: u64,
}
#[repr(C)]
pub(crate) struct CudaReductionTask {
    pub(crate) reduced: *mut c_void,
    pub(crate) lde: *const c_void,
    pub(crate) height: usize,
    pub(crate) inv_offset: usize,
    pub(crate) y0: u64,
    pub(crate) y1: u64,
    pub(crate) offset0: u64,
    pub(crate) offset1: u64,
}

pub(crate) struct CudaFriWorkspace {
    device_id: i32,
    handle: NonNull<c_void>,
}
impl CudaFriWorkspace {
    pub(crate) fn new(
        device_id: i32,
        points: &[[Goldilocks; 2]],
        counts: &[usize],
        coset: &[Goldilocks],
        ext_w: Goldilocks,
    ) -> Self {
        assert_eq!(points.len(), counts.len());
        let mut handle = core::ptr::null_mut();
        let status = unsafe {
            multi_stark_cuda_fri_workspace_create(
                device_id,
                &mut handle,
                points.as_ptr().cast(),
                counts.as_ptr(),
                points.len(),
                coset.as_ptr().cast(),
                coset.len(),
                raw_u64(ext_w),
            )
        };
        check_cuda(status, "FRI workspace allocation");
        Self {
            device_id,
            handle: NonNull::new(handle).unwrap(),
        }
    }
    pub(crate) fn interpolate(
        &mut self,
        tasks: &[CudaInterpolationTask],
        output_count: usize,
        ext_w: Goldilocks,
    ) -> Vec<[Goldilocks; 2]> {
        let mut output = vec![[Goldilocks::ZERO; 2]; output_count];
        let status = unsafe {
            multi_stark_cuda_fri_interpolate_batch(
                self.device_id,
                self.handle.as_ptr(),
                output.as_mut_ptr().cast(),
                output_count,
                tasks.as_ptr().cast(),
                tasks.len(),
                raw_u64(ext_w),
            )
        };
        check_cuda(status, "batched FRI interpolation");
        output
    }

    pub(crate) fn reduce(
        &mut self,
        tasks: &[CudaReductionTask],
        alpha: &[[Goldilocks; 2]],
        ext_w: Goldilocks,
    ) {
        let status = unsafe {
            multi_stark_cuda_fri_reduce_batch(
                self.device_id,
                self.handle.as_ptr(),
                tasks.as_ptr().cast(),
                tasks.len(),
                alpha.as_ptr().cast(),
                alpha.len(),
                raw_u64(ext_w),
            )
        };
        check_cuda(status, "batched FRI reduction")
    }
}
impl Drop for CudaFriWorkspace {
    fn drop(&mut self) {
        unsafe {
            let _ = multi_stark_cuda_fri_workspace_destroy(self.device_id, self.handle.as_ptr());
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn lookup_lde_resident(
    dft: &CudaDft,
    multiplicities: &[Goldilocks],
    args: &[Goldilocks],
    arg_offsets: &[usize],
    height: usize,
    num_lookups: usize,
    group_size: usize,
    beta: [Goldilocks; 2],
    gamma: [Goldilocks; 2],
    ext_w: Goldilocks,
    log_blowup: usize,
) -> (CudaLde, [Goldilocks; 2]) {
    assert!((1..=8).contains(&group_size));
    assert_eq!(arg_offsets.len(), num_lookups + 1);
    assert_eq!(arg_offsets.first(), Some(&0));
    assert!(arg_offsets.windows(2).all(|pair| pair[0] <= pair[1]));
    if num_lookups == 0 {
        let extended_height = height << log_blowup;
        let mut handle = core::ptr::null_mut();
        let status = unsafe {
            multi_stark_cuda_zero_lde_create(dft.device_id, &mut handle, extended_height, 2)
        };
        check_cuda(status, "zero lookup LDE");
        return (
            CudaLde {
                device_id: dft.device_id,
                handle: NonNull::new(handle).expect("CUDA returned a null zero LDE"),
                height: extended_height,
                width: 2,
            },
            [Goldilocks::ZERO; 2],
        );
    }
    let slots = num_lookups.div_ceil(group_size.max(1));
    let args_width = *arg_offsets.last().unwrap();
    assert_eq!(multiplicities.len(), height * num_lookups);
    assert_eq!(args.len(), height * args_width);
    let extended_height = height << log_blowup;
    let inverse_twiddles = dft.twiddles(log2_strict_usize(height), true);
    let shift_powers = dft.shift_powers(height, Goldilocks::GENERATOR);
    let forward_twiddles = dft.twiddles(log2_strict_usize(extended_height), false);
    let height_inverse = Goldilocks::ONE.div_2exp_u64(log2_strict_usize(height) as u64);
    let mut tail = [Goldilocks::ZERO; 4];
    let mut handle = core::ptr::null_mut();
    let status = unsafe {
        multi_stark_cuda_lookup_lde(
            dft.device_id,
            &mut handle,
            tail.as_mut_ptr().cast(),
            multiplicities.as_ptr().cast(),
            args.as_ptr().cast(),
            arg_offsets.as_ptr(),
            height,
            num_lookups,
            args_width,
            group_size.max(1),
            beta.as_ptr().cast(),
            gamma.as_ptr().cast(),
            raw_u64(ext_w),
            log_blowup,
            inverse_twiddles.as_ptr().cast(),
            shift_powers.as_ptr().cast(),
            forward_twiddles.as_ptr().cast(),
            raw_u64(height_inverse),
        )
    };
    check_cuda(status, "resident CUDA lookup LDE");
    (
        CudaLde {
            device_id: dft.device_id,
            handle: NonNull::new(handle).expect("CUDA returned a null lookup LDE"),
            height: extended_height,
            width: 2 * slots,
        },
        [tail[0] + tail[2], tail[1] + tail[3]],
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn lookup_lde_resident_partitioned(
    dft: &CudaDft,
    multiplicities: &[Goldilocks],
    args: &[Goldilocks],
    arg_offsets: &[usize],
    height: usize,
    num_lookups: usize,
    group_size: usize,
    beta: [Goldilocks; 2],
    gamma: [Goldilocks; 2],
    ext_w: Goldilocks,
    log_blowup: usize,
    cpu_deltas: impl Fn(core::ops::Range<usize>) -> Vec<[Goldilocks; 2]> + Sync,
) -> (CudaLde, [Goldilocks; 2]) {
    assert!((1..=8).contains(&group_size));
    assert_eq!(arg_offsets.len(), num_lookups + 1);
    assert_eq!(arg_offsets.first(), Some(&0));
    assert!(arg_offsets.windows(2).all(|pair| pair[0] <= pair[1]));
    assert!(num_lookups != 0);
    let slots = num_lookups.div_ceil(group_size);
    let args_width = *arg_offsets.last().unwrap();
    assert_eq!(multiplicities.len(), height * num_lookups);
    assert_eq!(args.len(), height * args_width);
    let extended_height = height << log_blowup;
    let inverse_twiddles = dft.twiddles(log2_strict_usize(height), true);
    let shift_powers = dft.shift_powers(height, Goldilocks::GENERATOR);
    let forward_twiddles = dft.twiddles(log2_strict_usize(extended_height), false);
    let height_inverse = Goldilocks::ONE.div_2exp_u64(log2_strict_usize(height) as u64);
    let total_started = std::time::Instant::now();
    let create_started = std::time::Instant::now();
    let mut pending_handle = core::ptr::null_mut();
    let status = unsafe {
        multi_stark_cuda_lookup_lde_begin_partitioned(
            dft.device_id,
            &mut pending_handle,
            arg_offsets.as_ptr(),
            height,
            num_lookups,
            args_width,
            group_size,
            log_blowup,
        )
    };
    check_cuda(status, "create partitioned CUDA lookup");
    let create_elapsed = create_started.elapsed();
    let pending_handle =
        NonNull::new(pending_handle).expect("CUDA returned a null partitioned lookup handle");
    struct PendingGuard {
        device_id: i32,
        handle: Option<NonNull<c_void>>,
    }
    impl Drop for PendingGuard {
        fn drop(&mut self) {
            if let Some(handle) = self.handle {
                unsafe {
                    let _ = multi_stark_cuda_lookup_lde_cancel_partitioned(
                        self.device_id,
                        handle.as_ptr(),
                    );
                }
            }
        }
    }
    let mut pending = PendingGuard {
        device_id: dft.device_id,
        handle: Some(pending_handle),
    };

    const ROWS_PER_CHUNK: usize = 1 << 16;
    let chunk_count = height.div_ceil(ROWS_PER_CHUNK);
    let remaining = std::sync::Mutex::new((0usize, chunk_count));
    let (cpu_chunks, cpu_elapsed, gpu_chunks, gpu_elapsed) = std::thread::scope(|scope| {
        let raw_pending = pending_handle.as_ptr() as usize;
        let gpu_remaining = &remaining;
        let device_id = dft.device_id;
        let ext_w = raw_u64(ext_w);
        let gpu_started = std::time::Instant::now();
        let gpu_worker = scope.spawn(move || {
            let mut chunks = 0usize;
            loop {
                let chunk = {
                    let mut remaining = gpu_remaining.lock().expect("lookup scheduler poisoned");
                    if remaining.0 == remaining.1 {
                        None
                    } else {
                        remaining.1 -= 1;
                        Some(remaining.1)
                    }
                };
                let Some(chunk) = chunk else { break };
                let row_start = chunk * ROWS_PER_CHUNK;
                let rows = (height - row_start).min(ROWS_PER_CHUNK);
                let status = unsafe {
                    multi_stark_cuda_lookup_lde_gpu_rows_partitioned(
                        device_id,
                        raw_pending as *mut c_void,
                        multiplicities.as_ptr().cast(),
                        args.as_ptr().cast(),
                        row_start,
                        rows,
                        beta.as_ptr().cast(),
                        gamma.as_ptr().cast(),
                        ext_w,
                    )
                };
                check_cuda(status, "partitioned CUDA lookup chunk");
                chunks += 1;
            }
            (chunks, gpu_started.elapsed())
        });

        let cpu_started = std::time::Instant::now();
        let mut cpu_chunks = 0usize;
        loop {
            let chunk = {
                let mut remaining = remaining.lock().expect("lookup scheduler poisoned");
                if remaining.0 == remaining.1 {
                    None
                } else {
                    let chunk = remaining.0;
                    remaining.0 += 1;
                    Some(chunk)
                }
            };
            let Some(chunk) = chunk else { break };
            let row_start = chunk * ROWS_PER_CHUNK;
            let rows = (height - row_start).min(ROWS_PER_CHUNK);
            let values = cpu_deltas(row_start..row_start + rows);
            assert_eq!(values.len(), rows * slots);
            let status = unsafe {
                multi_stark_cuda_lookup_lde_cpu_rows_partitioned(
                    dft.device_id,
                    pending_handle.as_ptr(),
                    values.as_ptr().cast(),
                    row_start,
                    rows,
                )
            };
            check_cuda(status, "upload partitioned CPU lookup chunk");
            cpu_chunks += 1;
        }
        let cpu_elapsed = cpu_started.elapsed();
        let (gpu_chunks, gpu_elapsed) = gpu_worker
            .join()
            .expect("partitioned GPU lookup worker panicked");
        (cpu_chunks, cpu_elapsed, gpu_chunks, gpu_elapsed)
    });

    let finish_started = std::time::Instant::now();
    let mut handle = core::ptr::null_mut();
    let mut tail = [Goldilocks::ZERO; 4];
    let status = unsafe {
        multi_stark_cuda_lookup_lde_finish_partitioned(
            dft.device_id,
            pending_handle.as_ptr(),
            &mut handle,
            tail.as_mut_ptr().cast(),
            inverse_twiddles.as_ptr().cast(),
            shift_powers.as_ptr().cast(),
            forward_twiddles.as_ptr().cast(),
            raw_u64(height_inverse),
        )
    };
    pending.handle = None;
    check_cuda(status, "partitioned CUDA lookup finish");
    let result = (
        CudaLde {
            device_id: dft.device_id,
            handle: NonNull::new(handle).expect("CUDA returned a null lookup LDE"),
            height: extended_height,
            width: 2 * slots,
        },
        [tail[0] + tail[2], tail[1] + tail[3]],
    );
    if memory_diagnostics_enabled() {
        eprintln!(
            "[multi-stark/cuda] cooperative lookup: cpu_chunks={cpu_chunks}/{chunk_count} cpu={:.3}s gpu_chunks={gpu_chunks}/{chunk_count} gpu={:.3}s create={:.3}s finish={:.3}s total={:.3}s",
            cpu_elapsed.as_secs_f64(),
            gpu_elapsed.as_secs_f64(),
            create_elapsed.as_secs_f64(),
            finish_started.elapsed().as_secs_f64(),
            total_started.elapsed().as_secs_f64()
        );
    }
    result
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn lookup_graph_lde_resident(
    dft: &CudaDft,
    graph: &ConstraintGraph<Goldilocks>,
    preprocessed: Option<&CudaLde>,
    main: &CudaLde,
    height: usize,
    group_size: usize,
    beta: [Goldilocks; 2],
    gamma: [Goldilocks; 2],
    ext_w: Goldilocks,
    log_blowup: usize,
) -> Option<(CudaLde, [Goldilocks; 2])> {
    assert!((1..=8).contains(&group_size));
    let (nodes, slot_count, lookups, args) = encode_lookup_nodes(graph)?;
    let num_lookups = lookups.len();
    let groups = num_lookups.div_ceil(group_size.max(1));
    let extended_height = height << log_blowup;
    let inverse_twiddles = dft.twiddles(log2_strict_usize(height), true);
    let shift_powers = dft.shift_powers(height, Goldilocks::GENERATOR);
    let forward_twiddles = dft.twiddles(log2_strict_usize(extended_height), false);
    let height_inverse = Goldilocks::ONE.div_2exp_u64(log2_strict_usize(height) as u64);
    let mut tail = [Goldilocks::ZERO; 4];
    let mut handle = core::ptr::null_mut();
    let status = unsafe {
        multi_stark_cuda_lookup_graph_lde(
            dft.device_id,
            &mut handle,
            tail.as_mut_ptr().cast(),
            nodes.as_ptr().cast(),
            nodes.len(),
            slot_count,
            lookups.as_ptr().cast(),
            lookups.len(),
            args.as_ptr(),
            args.len(),
            preprocessed.map_or(core::ptr::null(), CudaLde::raw_handle),
            main.raw_handle(),
            group_size.max(1),
            beta.as_ptr().cast(),
            gamma.as_ptr().cast(),
            raw_u64(ext_w),
            log_blowup,
            inverse_twiddles.as_ptr().cast(),
            shift_powers.as_ptr().cast(),
            forward_twiddles.as_ptr().cast(),
            raw_u64(height_inverse),
        )
    };
    check_cuda(status, "resident CUDA graph lookup LDE");
    Some((
        CudaLde {
            device_id: dft.device_id,
            handle: NonNull::new(handle).expect("CUDA returned a null graph lookup LDE"),
            height: extended_height,
            width: 2 * groups,
        },
        [tail[0] + tail[2], tail[1] + tail[3]],
    ))
}

impl CudaLde {
    pub(crate) fn interpolation_task(
        &self,
        height: usize,
        inv_offset: usize,
        output_offset: usize,
        scale: [Goldilocks; 2],
    ) -> CudaInterpolationTask {
        CudaInterpolationTask {
            lde: self.raw_handle(),
            height,
            inv_offset,
            output_offset,
            scale0: raw_u64(scale[0]),
            scale1: raw_u64(scale[1]),
        }
    }
}
impl CudaReducedOpening {
    pub(crate) fn reduction_task(
        &mut self,
        lde: &CudaLde,
        inv_offset: usize,
        y: [Goldilocks; 2],
        offset: [Goldilocks; 2],
    ) -> CudaReductionTask {
        CudaReductionTask {
            reduced: self.handle.as_ptr(),
            lde: lde.raw_handle(),
            height: self.height,
            inv_offset,
            y0: raw_u64(y[0]),
            y1: raw_u64(y[1]),
            offset0: raw_u64(offset[0]),
            offset1: raw_u64(offset[1]),
        }
    }
}

impl Drop for CudaLde {
    fn drop(&mut self) {
        // SAFETY: this wrapper uniquely owns the handle and drops it once.
        unsafe {
            let _ = multi_stark_cuda_lde_destroy(self.device_id, self.handle.as_ptr());
        }
    }
}

impl TwoAdicSubgroupDft<Goldilocks> for CudaDft {
    type Evaluations = BitReversedMatrixView<RowMajorMatrix<Goldilocks>>;

    fn dft_batch(&self, mut matrix: RowMajorMatrix<Goldilocks>) -> Self::Evaluations {
        let height = matrix.height();
        let width = matrix.width();
        Self::validate_dimensions(height, width);
        if height == 1 || width == 0 {
            return BitReversalPerm::new_view(matrix);
        }
        if !Self::use_cuda_dft(height, width) {
            return self.cpu.dft_batch(matrix);
        }

        let twiddles = self.twiddles(log2_strict_usize(height), false);
        // SAFETY: Goldilocks is repr(transparent) over u64 (asserted above),
        // every u64 bit pattern is a valid Goldilocks value, all buffers have
        // the element counts implied by height/width, and the FFI call is
        // synchronous so the borrowed twiddle buffer outlives device use.
        let status = unsafe {
            multi_stark_cuda_dft_batch(
                self.device_id,
                matrix.values.as_mut_ptr().cast(),
                height,
                width,
                twiddles.as_ptr().cast(),
            )
        };
        check_cuda(status, "batched DFT");

        // The CUDA DIF kernel writes bit-reversed rows. Wrap that storage so
        // callers observe the natural-order evaluations required by the trait.
        BitReversalPerm::new_view(matrix)
    }

    fn coset_lde_batch(
        &self,
        matrix: RowMajorMatrix<Goldilocks>,
        added_bits: usize,
        shift: Goldilocks,
    ) -> Self::Evaluations {
        let height = matrix.height();
        let width = matrix.width();
        Self::validate_dimensions(height, width);
        let extended_height = height
            .checked_shl(u32::try_from(added_bits).expect("LDE blowup exceeds u32"))
            .expect("LDE height overflows usize");
        Self::validate_dimensions(extended_height, width);

        if width == 0 {
            return BitReversalPerm::new_view(RowMajorMatrix::new(Vec::new(), width));
        }
        if height == 1 {
            let mut values = Vec::with_capacity(extended_height * width);
            for _ in 0..extended_height {
                values.extend_from_slice(&matrix.values);
            }
            return BitReversalPerm::new_view(RowMajorMatrix::new(values, width));
        }
        if !Self::use_cuda_coset_lde(extended_height, width) {
            return self.cpu.coset_lde_batch(matrix, added_bits, shift);
        }

        let log_height = log2_strict_usize(height);
        let inverse_twiddles = self.twiddles(log_height, true);
        let forward_twiddles = self.twiddles(log2_strict_usize(extended_height), false);
        let shift_powers = self.shift_powers(height, shift);
        let height_inverse = Goldilocks::ONE.div_2exp_u64(log_height as u64);
        let mut output = Goldilocks::zero_vec(extended_height * width);

        // SAFETY: the input/output and cached tables have the exact lengths
        // implied by the dimensions, their Goldilocks storage is u64-compatible,
        // and the FFI call completes before any borrowed buffer is released.
        let status = unsafe {
            multi_stark_cuda_coset_lde_batch(
                self.device_id,
                output.as_mut_ptr().cast(),
                matrix.values.as_ptr().cast(),
                height,
                width,
                added_bits,
                inverse_twiddles.as_ptr().cast(),
                shift_powers.as_ptr().cast(),
                forward_twiddles.as_ptr().cast(),
                raw_u64(height_inverse),
            )
        };
        check_cuda(status, "coset LDE");

        BitReversalPerm::new_view(RowMajorMatrix::new(output, width))
    }
}

#[inline]
fn raw_u64(value: Goldilocks) -> u64 {
    value.as_canonical_u64()
}

pub(crate) fn device_memory_info(device_id: i32) -> (usize, usize) {
    let mut free_bytes = 0;
    let mut total_bytes = 0;
    let status =
        unsafe { multi_stark_cuda_memory_info(device_id, &mut free_bytes, &mut total_bytes) };
    check_cuda(status, "CUDA device initialization");
    (free_bytes, total_bytes)
}

pub(crate) fn memory_diagnostics_enabled() -> bool {
    std::env::var_os("MULTI_STARK_CUDA_MEMORY_LOG").is_some()
}

fn check_cuda(status: i32, operation: &str) {
    if status == 0 {
        return;
    }
    // SAFETY: CUDA returns a pointer to a process-lifetime, NUL-terminated
    // static error string for every status code.
    let message = unsafe {
        let pointer = multi_stark_cuda_error_string(status);
        if pointer.is_null() {
            "unknown CUDA error".into()
        } else {
            CStr::from_ptr(pointer).to_string_lossy()
        }
    };
    panic!("CUDA {operation} failed ({status}): {message}");
}

/// Hashes fixed-width byte rows with the first-party CUDA BLAKE3 kernel.
///
/// This is the byte-compatible leaf/node primitive used by the resident
/// Merkle backend. Messages may span up to 32 BLAKE3 chunks (32 KiB).
#[must_use]
pub fn blake3_hash_rows(device_id: i32, messages: &[u8], message_bytes: usize) -> Vec<[u8; 32]> {
    assert!(device_id >= 0, "CUDA device id must be non-negative");
    assert!(message_bytes != 0, "BLAKE3 rows must not be empty");
    assert!(message_bytes <= 32 * 1024, "BLAKE3 rows exceed 32 KiB");
    assert!(
        messages.len().is_multiple_of(message_bytes),
        "message buffer is not a whole number of rows"
    );
    let message_count = messages.len() / message_bytes;
    if message_count == 0 {
        return Vec::new();
    }
    let mut digests = vec![[0u8; 32]; message_count];
    // SAFETY: the input has `message_count` complete fixed-width rows, the
    // output has one 32-byte digest per row, and the call is synchronous.
    let status = unsafe {
        multi_stark_cuda_blake3_hash_rows(
            device_id,
            digests.as_mut_ptr().cast(),
            messages.as_ptr(),
            message_bytes,
            message_count,
        )
    };
    check_cuda(status, "BLAKE3 row hashing");
    digests
}

/// Builds a binary BLAKE3 Merkle tree for fixed-width rows on the GPU and
/// returns its root. Intermediate digest layers remain device-resident.
#[must_use]
#[cfg(test)]
pub(crate) fn blake3_merkle_root(device_id: i32, rows: &[u8], row_bytes: usize) -> [u8; 32] {
    assert!(device_id >= 0, "CUDA device id must be non-negative");
    assert!(row_bytes != 0, "Merkle rows must not be empty");
    assert!(
        rows.len().is_multiple_of(row_bytes),
        "row buffer is not a whole number of rows"
    );
    let row_count = rows.len() / row_bytes;
    assert!(
        row_count.is_power_of_two(),
        "Merkle row count must be a power of two"
    );
    let mut root = [0u8; 32];
    // SAFETY: `rows` contains `row_count` fixed-width rows, `root` has the
    // required 32 bytes, and the call is synchronous.
    let status = unsafe {
        multi_stark_cuda_blake3_merkle_root(
            device_id,
            root.as_mut_ptr(),
            rows.as_ptr(),
            row_bytes,
            row_count,
        )
    };
    check_cuda(status, "BLAKE3 Merkle root");
    root
}

/// Device-resident binary BLAKE3 Merkle tree and its fixed-width source rows.
#[cfg(test)]
pub(crate) struct CudaMerkleTree {
    device_id: i32,
    handle: NonNull<c_void>,
    root: [u8; 32],
    row_bytes: usize,
    row_count: usize,
}

// CUDA allocations are process resources rather than thread-affine handles;
// every operation selects `device_id` before touching the allocation.
#[cfg(test)]
unsafe impl Send for CudaMerkleTree {}

#[cfg(test)]
impl CudaMerkleTree {
    /// Uploads rows once and builds every digest layer on the selected GPU.
    #[must_use]
    pub(crate) fn new(device_id: i32, rows: &[u8], row_bytes: usize) -> Self {
        assert!(device_id >= 0, "CUDA device id must be non-negative");
        assert!(row_bytes != 0, "Merkle rows must not be empty");
        assert!(
            rows.len().is_multiple_of(row_bytes),
            "row buffer is not a whole number of rows"
        );
        let row_count = rows.len() / row_bytes;
        assert!(
            row_count.is_power_of_two(),
            "Merkle row count must be a power of two"
        );
        let mut handle = core::ptr::null_mut();
        let mut root = [0u8; 32];
        // SAFETY: input dimensions were checked, the output pointers are valid,
        // and successful creation transfers ownership to this RAII wrapper.
        let status = unsafe {
            multi_stark_cuda_merkle_create(
                device_id,
                &mut handle,
                root.as_mut_ptr(),
                rows.as_ptr(),
                row_bytes,
                row_count,
            )
        };
        check_cuda(status, "resident Merkle tree creation");
        Self {
            device_id,
            handle: NonNull::new(handle).expect("CUDA returned a null Merkle handle"),
            root,
            row_bytes,
            row_count,
        }
    }

    /// Returns the BLAKE3 root copied out during tree construction.
    #[must_use]
    pub(crate) const fn root(&self) -> [u8; 32] {
        self.root
    }

    /// Copies one queried row and its bottom-up authentication siblings.
    #[must_use]
    pub(crate) fn open(&self, index: usize) -> (Vec<u8>, Vec<[u8; 32]>) {
        assert!(index < self.row_count, "Merkle opening index out of bounds");
        let mut row = vec![0u8; self.row_bytes];
        let mut siblings = vec![[0u8; 32]; self.row_count.trailing_zeros() as usize];
        // SAFETY: the handle remains owned by `self`; all output allocations
        // have the exact sizes implied by the resident tree dimensions.
        let status = unsafe {
            multi_stark_cuda_merkle_open(
                self.device_id,
                self.handle.as_ptr(),
                index,
                row.as_mut_ptr(),
                siblings.as_mut_ptr().cast(),
            )
        };
        check_cuda(status, "resident Merkle opening");
        (row, siblings)
    }
}

#[cfg(test)]
impl Drop for CudaMerkleTree {
    fn drop(&mut self) {
        // SAFETY: this wrapper uniquely owns the handle and drops it once.
        unsafe {
            let _ = multi_stark_cuda_merkle_destroy(self.device_id, self.handle.as_ptr());
        }
    }
}

/// Device-resident digest tree for Plonky3's mixed-height binary MMCS layout.
#[doc(hidden)]
pub struct CudaMixedMerkleTree {
    device_id: i32,
    handle: NonNull<c_void>,
    root: [u8; 32],
    row_count: usize,
}

unsafe impl Send for CudaMixedMerkleTree {}
unsafe impl Sync for CudaMixedMerkleTree {}

impl CudaMixedMerkleTree {
    #[must_use]
    pub(crate) fn hash_hybrid_height_group(
        device_id: i32,
        ldes: &[Option<&CudaLde>],
        host_matrices: &[Option<&RowMajorMatrix<Goldilocks>>],
    ) -> Vec<[u8; 32]> {
        assert!(device_id >= 0, "CUDA device id must be non-negative");
        assert_eq!(ldes.len(), host_matrices.len());
        assert!(!ldes.is_empty(), "hybrid height group is empty");
        assert!(
            ldes.iter().flatten().all(|lde| lde.device_id == device_id),
            "hybrid height group spans CUDA devices"
        );
        assert!(
            ldes.iter()
                .zip(host_matrices)
                .all(|(lde, matrix)| lde.is_some() ^ matrix.is_some()),
            "every hybrid height-group matrix must have exactly one storage backend"
        );
        let heights = ldes
            .iter()
            .zip(host_matrices)
            .map(|(lde, matrix)| lde.map_or_else(|| matrix.unwrap().height(), CudaLde::height))
            .collect::<Vec<_>>();
        let height = heights[0];
        assert!(
            heights.iter().all(|&candidate| candidate == height),
            "hybrid height group contains unequal heights"
        );
        let handles = ldes
            .iter()
            .map(|lde| lde.map_or(core::ptr::null(), CudaLde::raw_handle))
            .collect::<Vec<_>>();
        let host_values = host_matrices
            .iter()
            .map(|matrix| matrix.map_or(core::ptr::null(), |matrix| matrix.values.as_ptr().cast()))
            .collect::<Vec<_>>();
        let widths = ldes
            .iter()
            .zip(host_matrices)
            .map(|(lde, matrix)| lde.map_or_else(|| matrix.unwrap().width(), CudaLde::width))
            .collect::<Vec<_>>();
        let mut digests = vec![[0u8; 32]; height];
        // SAFETY: every pointer is backed by a matrix or CUDA handle borrowed
        // for this synchronous call, and `digests` has exactly `height` rows.
        let status = unsafe {
            multi_stark_cuda_hash_hybrid_height_group(
                device_id,
                digests.as_mut_ptr().cast(),
                handles.as_ptr(),
                host_values.as_ptr(),
                widths.as_ptr(),
                heights.as_ptr(),
                handles.len(),
                height,
            )
        };
        check_cuda(status, "hybrid CUDA height-group hashing");
        digests
    }

    /// Builds a mixed-height tree from row groups ordered by descending,
    /// distinct power-of-two height. Each group contains already-concatenated
    /// rows for all matrices entering the MMCS at that height.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn new(device_id: i32, levels: &[(&[u8], usize, usize)]) -> Self {
        assert!(device_id >= 0, "CUDA device id must be non-negative");
        assert!(!levels.is_empty(), "mixed Merkle tree has no row groups");
        for (index, (rows, row_bytes, height)) in levels.iter().copied().enumerate() {
            assert!(row_bytes != 0, "mixed Merkle rows must not be empty");
            assert_eq!(
                rows.len(),
                row_bytes * height,
                "mixed Merkle row-group dimensions disagree"
            );
            assert!(
                height.is_power_of_two(),
                "mixed Merkle height is not a power of two"
            );
            if index > 0 {
                assert!(
                    height < levels[index - 1].2,
                    "mixed Merkle heights are not descending"
                );
            }
        }

        let row_pointers: Vec<*const u8> =
            levels.iter().map(|(rows, _, _)| rows.as_ptr()).collect();
        let row_bytes: Vec<usize> = levels.iter().map(|(_, width, _)| *width).collect();
        let heights: Vec<usize> = levels.iter().map(|(_, _, height)| *height).collect();
        let mut handle = core::ptr::null_mut();
        let mut root = [0u8; 32];
        // SAFETY: every level buffer and dimension was validated above; the
        // call copies all host inputs synchronously and returns an owned handle.
        let status = unsafe {
            multi_stark_cuda_mixed_merkle_create(
                device_id,
                &mut handle,
                root.as_mut_ptr(),
                row_pointers.as_ptr(),
                row_bytes.as_ptr(),
                heights.as_ptr(),
                levels.len(),
            )
        };
        check_cuda(status, "resident mixed-height Merkle tree creation");
        Self {
            device_id,
            handle: NonNull::new(handle).expect("CUDA returned a null mixed Merkle handle"),
            root,
            row_count: heights[0],
        }
    }

    /// Builds the exact mixed-height tree directly from resident bit-reversed
    /// LDE matrices. Matrix rows are concatenated in slice order at each
    /// height using device-to-device copies.
    #[must_use]
    pub(crate) fn from_ldes(device_id: i32, ldes: &[CudaLde]) -> Self {
        assert!(device_id >= 0, "CUDA device id must be non-negative");
        assert!(!ldes.is_empty(), "mixed Merkle tree has no resident LDEs");
        assert!(
            ldes.iter().all(|lde| lde.device_id == device_id),
            "resident LDEs belong to different CUDA devices"
        );
        let handles: Vec<*const c_void> = ldes
            .iter()
            .map(|lde| lde.handle.as_ptr().cast_const())
            .collect();
        let mut handle = core::ptr::null_mut();
        let mut root = [0u8; 32];
        // SAFETY: every handle is live for the synchronous construction call;
        // the resulting tree owns only its own digest allocation.
        let status = unsafe {
            multi_stark_cuda_mixed_merkle_create_from_ldes(
                device_id,
                &mut handle,
                root.as_mut_ptr(),
                handles.as_ptr(),
                handles.len(),
            )
        };
        check_cuda(status, "resident LDE Merkle tree creation");
        Self {
            device_id,
            handle: NonNull::new(handle).expect("CUDA returned a null mixed Merkle handle"),
            root,
            row_count: ldes.iter().map(CudaLde::height).max().unwrap(),
        }
    }

    /// Builds a mixed-height tree from resident CUDA LDEs, host matrices, and
    /// pre-hashed height groups. Deferred dimensions are accepted only when a
    /// pre-hashed group supplies that height's complete digest frontier.
    #[must_use]
    pub(crate) fn from_hybrid(
        device_id: i32,
        ldes: &[Option<&CudaLde>],
        host_matrices: &[Option<&RowMajorMatrix<Goldilocks>>],
        deferred_dimensions: &[Option<Dimensions>],
        host_digest_groups: &[(usize, Vec<[u8; 32]>)],
    ) -> Self {
        assert!(device_id >= 0, "CUDA device id must be non-negative");
        assert_eq!(ldes.len(), host_matrices.len());
        assert_eq!(ldes.len(), deferred_dimensions.len());
        assert!(!ldes.is_empty(), "mixed Merkle tree has no LDE groups");
        assert!(
            ldes.iter().flatten().all(|lde| lde.device_id == device_id),
            "resident LDEs belong to different CUDA devices"
        );
        for (height, digests) in host_digest_groups {
            assert!(
                height.is_power_of_two(),
                "host digest height is not a power of two"
            );
            assert_eq!(
                *height,
                digests.len(),
                "host digest group has the wrong height"
            );
        }
        assert!(
            ldes.iter().zip(host_matrices).zip(deferred_dimensions).all(
                |((lde, matrix), deferred)| {
                    let sources = usize::from(lde.is_some())
                        + usize::from(matrix.is_some())
                        + usize::from(deferred.is_some());
                    sources == 1
                        && deferred.is_none_or(|dimensions| {
                            host_digest_groups
                                .iter()
                                .any(|(height, _)| *height == dimensions.height)
                        })
                }
            ),
            "every hybrid matrix needs one source, and deferred matrices need pre-hashed rows"
        );
        let handles: Vec<*const c_void> = ldes
            .iter()
            .map(|lde| lde.map_or(core::ptr::null(), |lde| lde.handle.as_ptr().cast_const()))
            .collect();
        let host_values = host_matrices
            .iter()
            .map(|matrix| {
                matrix.map_or(core::ptr::null(), |matrix| {
                    matrix.values.as_ptr().cast::<u64>()
                })
            })
            .collect::<Vec<_>>();
        let widths = ldes
            .iter()
            .zip(host_matrices)
            .zip(deferred_dimensions)
            .map(|((lde, matrix), deferred)| {
                lde.map_or_else(
                    || matrix.map_or_else(|| deferred.unwrap().width, Matrix::width),
                    CudaLde::width,
                )
            })
            .collect::<Vec<_>>();
        let heights: Vec<usize> = ldes
            .iter()
            .zip(host_matrices)
            .zip(deferred_dimensions)
            .map(|((lde, matrix), deferred)| {
                lde.map_or_else(
                    || matrix.map_or_else(|| deferred.unwrap().height, Matrix::height),
                    CudaLde::height,
                )
            })
            .collect();
        let host_digest_pointers: Vec<*const u8> = host_digest_groups
            .iter()
            .map(|(_, digests)| digests.as_ptr().cast())
            .collect();
        let host_digest_heights: Vec<usize> = host_digest_groups
            .iter()
            .map(|(height, _)| *height)
            .collect();
        let mut handle = core::ptr::null_mut();
        let mut root = [0u8; 32];
        // SAFETY: all resident handles and host digest buffers remain live for
        // this synchronous call; the returned tree owns its digest allocation.
        let status = unsafe {
            multi_stark_cuda_mixed_merkle_create_hybrid(
                device_id,
                &mut handle,
                root.as_mut_ptr(),
                handles.as_ptr(),
                host_values.as_ptr(),
                widths.as_ptr(),
                heights.as_ptr(),
                ldes.len(),
                host_digest_pointers.as_ptr(),
                host_digest_heights.as_ptr(),
                host_digest_groups.len(),
            )
        };
        check_cuda(status, "hybrid CPU/CUDA mixed-height Merkle tree creation");
        let row_count = heights.into_iter().max().unwrap();
        Self {
            device_id,
            handle: NonNull::new(handle).expect("CUDA returned a null mixed Merkle handle"),
            root,
            row_count,
        }
    }

    pub(crate) fn from_fri_codeword(codeword: &CudaLde, arity: usize) -> Self {
        assert_eq!(codeword.width, 2);
        assert!(arity.is_power_of_two());
        assert_eq!(codeword.height % arity, 0);
        let mut handle = core::ptr::null_mut();
        let mut root = [0u8; 32];
        let status = unsafe {
            multi_stark_cuda_fri_merkle_create(
                codeword.device_id,
                &mut handle,
                root.as_mut_ptr(),
                codeword.raw_handle(),
                arity,
            )
        };
        check_cuda(status, "resident CUDA FRI Merkle tree creation");
        Self {
            device_id: codeword.device_id,
            handle: NonNull::new(handle).expect("CUDA returned a null FRI Merkle handle"),
            root,
            row_count: codeword.height / arity,
        }
    }

    /// Builds a mixed-height tree by uploading each host matrix directly,
    /// without constructing a concatenated host-side row buffer.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn from_host_matrices(
        device_id: i32,
        matrices: &[RowMajorMatrix<Goldilocks>],
    ) -> Self {
        assert!(device_id >= 0, "CUDA device id must be non-negative");
        assert!(!matrices.is_empty(), "mixed Merkle tree has no matrices");
        let values: Vec<*const u64> = matrices
            .iter()
            .map(|matrix| matrix.values.as_ptr().cast())
            .collect();
        let heights: Vec<_> = matrices.iter().map(Matrix::height).collect();
        let widths: Vec<_> = matrices.iter().map(Matrix::width).collect();
        let mut handle = core::ptr::null_mut();
        let mut root = [0u8; 32];
        // SAFETY: all matrix buffers remain live for the synchronous upload;
        // the returned tree owns only its device digest layers.
        let status = unsafe {
            multi_stark_cuda_mixed_merkle_create_from_host_matrices(
                device_id,
                &mut handle,
                root.as_mut_ptr(),
                values.as_ptr(),
                heights.as_ptr(),
                widths.as_ptr(),
                matrices.len(),
            )
        };
        check_cuda(status, "host-matrix CUDA Merkle tree creation");
        Self {
            device_id,
            handle: NonNull::new(handle).expect("CUDA returned a null mixed Merkle handle"),
            root,
            row_count: heights.into_iter().max().unwrap(),
        }
    }

    #[must_use]
    pub(crate) const fn root(&self) -> [u8; 32] {
        self.root
    }

    #[must_use]
    pub(crate) fn open_siblings(&self, index: usize) -> Vec<[u8; 32]> {
        assert!(
            index < self.row_count,
            "mixed Merkle opening index out of bounds"
        );
        let mut siblings = vec![[0u8; 32]; self.row_count.trailing_zeros() as usize];
        // SAFETY: the handle is live and the proof allocation has one sibling
        // for every binary layer.
        let status = unsafe {
            multi_stark_cuda_mixed_merkle_open(
                self.device_id,
                self.handle.as_ptr(),
                index,
                siblings.as_mut_ptr().cast(),
            )
        };
        check_cuda(status, "resident mixed-height Merkle opening");
        siblings
    }

    #[must_use]
    pub(crate) fn open_siblings_batch(&self, indices: &[usize]) -> Vec<Vec<[u8; 32]>> {
        assert!(!indices.is_empty(), "mixed Merkle opening batch is empty");
        assert!(
            indices.iter().all(|&index| index < self.row_count),
            "mixed Merkle opening index out of bounds"
        );
        let levels = self.row_count.trailing_zeros() as usize;
        if levels == 0 {
            return vec![Vec::new(); indices.len()];
        }
        let device_indices: Vec<u64> = indices
            .iter()
            .map(|&index| u64::try_from(index).expect("Merkle index exceeds u64"))
            .collect();
        let mut siblings = vec![[0u8; 32]; indices.len() * levels];
        let status = unsafe {
            multi_stark_cuda_mixed_merkle_open_batch(
                self.device_id,
                self.handle.as_ptr(),
                device_indices.as_ptr(),
                device_indices.len(),
                siblings.as_mut_ptr().cast(),
            )
        };
        check_cuda(status, "resident mixed-height Merkle opening batch");
        siblings
            .chunks_exact(levels)
            .map(<[[u8; 32]]>::to_vec)
            .collect()
    }

    /// Opens several leaves and prunes their overlapping binary authentication paths.
    ///
    /// The CUDA kernel returns one full path per requested index. Plonky3's multiproof
    /// format keeps only boundary digests, ordered by tree level and then parent index.
    #[must_use]
    pub(crate) fn open_pruned_siblings(&self, indices: &[usize]) -> PrunedMerklePaths<u8, 32> {
        if indices.is_empty() {
            return PrunedMerklePaths {
                sibling_hashes: Vec::new(),
            };
        }

        let paths = self.open_siblings_batch(indices);
        // Each frontier entry is (node index at this level, source query slot).
        // Sorting and deduplicating matches Plonky3's treatment of repeated queries.
        let mut frontier: Vec<_> = indices
            .iter()
            .copied()
            .enumerate()
            .map(|(slot, index)| (index, slot))
            .collect();
        frontier.sort_unstable_by_key(|&(index, _)| index);
        frontier.dedup_by_key(|entry| entry.0);

        let mut parents = Vec::with_capacity(frontier.len());
        let mut sibling_hashes = Vec::new();
        for (level, _) in paths[0].iter().enumerate() {
            parents.clear();
            let mut at = 0;
            while at < frontier.len() {
                let parent = frontier[at].0 >> 1;
                let lead_slot = frontier[at].1;
                let group_start = at;
                at += 1;
                while at < frontier.len() && frontier[at].0 >> 1 == parent {
                    at += 1;
                }

                let group_len = at - group_start;
                debug_assert!(group_len <= 2, "binary frontier group exceeds two children");
                if group_len == 1 {
                    sibling_hashes.push(paths[lead_slot][level]);
                }
                parents.push((parent, lead_slot));
            }
            core::mem::swap(&mut frontier, &mut parents);
        }

        PrunedMerklePaths { sibling_hashes }
    }
}

impl Drop for CudaMixedMerkleTree {
    fn drop(&mut self) {
        // SAFETY: this wrapper uniquely owns the handle and drops it once.
        unsafe {
            let _ = multi_stark_cuda_mixed_merkle_destroy(self.device_id, self.handle.as_ptr());
        }
    }
}

unsafe extern "C" {
    fn multi_stark_cuda_memory_info(
        device_id: i32,
        free_bytes: *mut usize,
        total_bytes: *mut usize,
    ) -> i32;

    fn multi_stark_cuda_dft_batch(
        device_id: i32,
        values: *mut u64,
        height: usize,
        width: usize,
        twiddles: *const u64,
    ) -> i32;

    fn multi_stark_cuda_coset_lde_batch(
        device_id: i32,
        output: *mut u64,
        input: *const u64,
        height: usize,
        width: usize,
        added_bits: usize,
        inverse_twiddles: *const u64,
        shift_powers: *const u64,
        forward_twiddles: *const u64,
        height_inverse: u64,
    ) -> i32;

    fn multi_stark_cuda_coset_lde_create(
        device_id: i32,
        handle: *mut *mut c_void,
        input: *const u64,
        height: usize,
        width: usize,
        added_bits: usize,
        inverse_twiddles: *const u64,
        shift_powers: *const u64,
        forward_twiddles: *const u64,
        height_inverse: u64,
    ) -> i32;

    fn multi_stark_cuda_prepare_lde_constants(
        device_id: i32,
        inverse_twiddles: *const u64,
        inverse_count: usize,
        shift_powers: *const u64,
        height: usize,
        forward_twiddles: *const u64,
        forward_count: usize,
    ) -> i32;

    fn multi_stark_cuda_lde_create_from_host(
        device_id: i32,
        handle: *mut *mut c_void,
        input: *const u64,
        height: usize,
        width: usize,
    ) -> i32;
    fn multi_stark_cuda_zero_lde_create(
        device_id: i32,
        handle: *mut *mut c_void,
        height: usize,
        width: usize,
    ) -> i32;

    fn multi_stark_cuda_lde_copy_to_host(
        device_id: i32,
        handle: *const c_void,
        output: *mut u64,
    ) -> i32;
    fn multi_stark_cuda_lde_release_trace(device_id: i32, handle: *mut c_void) -> i32;
    fn multi_stark_cuda_lde_release_values(device_id: i32, handle: *mut c_void) -> i32;
    fn multi_stark_cuda_lde_attach_trace(
        device_id: i32,
        handle: *mut c_void,
        trace: *const u64,
        height: usize,
        width: usize,
    ) -> i32;

    #[cfg(test)]
    fn multi_stark_cuda_lde_copy_row(
        device_id: i32,
        handle: *const c_void,
        row: usize,
        output: *mut u64,
    ) -> i32;

    fn multi_stark_cuda_lde_copy_rows(
        device_id: i32,
        handle: *const c_void,
        rows: *const u64,
        row_count: usize,
        output: *mut u64,
    ) -> i32;

    fn multi_stark_cuda_lde_destroy(device_id: i32, handle: *mut c_void) -> i32;

    #[cfg(test)]
    fn multi_stark_cuda_constraint_graph(
        device_id: i32,
        output: *mut u64,
        nodes: *const c_void,
        node_count: usize,
        roots: *const u32,
        root_count: usize,
        preprocessed_handle: *const c_void,
        main_handle: *const c_void,
        stage2_handle: *const c_void,
        publics: *const u64,
        public_count: usize,
        selectors: *const u64,
        quotient_size: usize,
        next_step: usize,
    ) -> i32;
    fn multi_stark_cuda_quotient_values(
        device_id: i32,
        output: *mut u64,
        nodes: *const c_void,
        node_count: usize,
        slot_count: usize,
        roots: *const u32,
        root_count: usize,
        lookups: *const c_void,
        lookup_count: usize,
        lookup_args: *const u32,
        lookup_arg_count: usize,
        group_size: usize,
        preprocessed_handle: *const c_void,
        main_handle: *const c_void,
        stage2_handle: *const c_void,
        publics: *const u64,
        public_count: usize,
        coset_shift: u64,
        coset_generator: u64,
        trace_last: u64,
        vanishing_start: u64,
        vanishing_step: u64,
        alpha: *const u64,
        constraint_count: usize,
        delta: *const u64,
        ext_w: u64,
        quotient_size: usize,
        next_step: usize,
    ) -> i32;
    fn multi_stark_cuda_quotient_lde(
        device_id: i32,
        output_handle: *mut *mut c_void,
        nodes: *const c_void,
        node_count: usize,
        slot_count: usize,
        roots: *const u32,
        root_count: usize,
        lookups: *const c_void,
        lookup_count: usize,
        lookup_args: *const u32,
        lookup_arg_count: usize,
        group_size: usize,
        preprocessed_handle: *const c_void,
        main_handle: *const c_void,
        stage2_handle: *const c_void,
        publics: *const u64,
        public_count: usize,
        coset_shift: u64,
        coset_generator: u64,
        trace_last: u64,
        vanishing_start: u64,
        vanishing_step: u64,
        alpha: *const u64,
        constraint_count: usize,
        delta: *const u64,
        ext_w: u64,
        quotient_size: usize,
        next_step: usize,
        quotient_degree: usize,
        log_blowup: usize,
        quotient_twiddles: *const u64,
        lde_twiddles: *const u64,
        slice_weights: *const u64,
    ) -> i32;
    fn multi_stark_cuda_quotient_lde_mixed(
        device_id: i32,
        output_handle: *mut *mut c_void,
        nodes: *const c_void,
        node_count: usize,
        slot_count: usize,
        roots: *const u32,
        root_count: usize,
        lookups: *const c_void,
        lookup_count: usize,
        lookup_args: *const u32,
        lookup_arg_count: usize,
        group_size: usize,
        preprocessed_handle: *const c_void,
        preprocessed_host: *const u64,
        preprocessed_height: usize,
        preprocessed_width: usize,
        main_handle: *const c_void,
        main_host: *const u64,
        main_height: usize,
        main_width: usize,
        stage2_handle: *const c_void,
        stage2_host: *const u64,
        stage2_height: usize,
        stage2_width: usize,
        publics: *const u64,
        public_count: usize,
        coset_shift: u64,
        coset_generator: u64,
        trace_last: u64,
        vanishing_start: u64,
        vanishing_step: u64,
        alpha: *const u64,
        constraint_count: usize,
        delta: *const u64,
        ext_w: u64,
        quotient_size: usize,
        next_step: usize,
        quotient_degree: usize,
        log_blowup: usize,
        quotient_twiddles: *const u64,
        lde_twiddles: *const u64,
        slice_weights: *const u64,
    ) -> i32;
    fn multi_stark_cuda_mixed_lde_open_row(
        device_id: i32,
        output: *mut u64,
        handles: *const *const c_void,
        handle_count: usize,
        max_height: usize,
        index: usize,
    ) -> i32;
    fn multi_stark_cuda_mixed_lde_open_rows(
        device_id: i32,
        output: *mut u64,
        handles: *const *const c_void,
        handle_count: usize,
        max_height: usize,
        indices: *const u64,
        query_count: usize,
    ) -> i32;
    #[cfg(test)]
    fn multi_stark_cuda_lde_interpolate(
        device_id: i32,
        output: *mut u64,
        handle: *const c_void,
        height: usize,
        inv_denoms: *const u64,
        coset: *const u64,
        scale: *const u64,
        ext_w: u64,
    ) -> i32;
    fn multi_stark_cuda_fri_workspace_create(
        device_id: i32,
        handle: *mut *mut c_void,
        points: *const u64,
        counts: *const usize,
        point_count: usize,
        coset: *const u64,
        coset_count: usize,
        ext_w: u64,
    ) -> i32;
    fn multi_stark_cuda_fri_interpolate_batch(
        device_id: i32,
        handle: *mut c_void,
        output: *mut u64,
        output_count: usize,
        tasks: *const c_void,
        task_count: usize,
        ext_w: u64,
    ) -> i32;
    fn multi_stark_cuda_fri_reduce_batch(
        device_id: i32,
        handle: *mut c_void,
        tasks: *const c_void,
        task_count: usize,
        alpha: *const u64,
        alpha_count: usize,
        ext_w: u64,
    ) -> i32;
    fn multi_stark_cuda_fri_workspace_destroy(device_id: i32, handle: *mut c_void) -> i32;
    fn multi_stark_cuda_reduced_into_lde(
        device_id: i32,
        output: *mut *mut c_void,
        reduced: *mut c_void,
    ) -> i32;
    fn multi_stark_cuda_fri_fold_resident(
        device_id: i32,
        output: *mut *mut c_void,
        input: *const c_void,
        next_reduced: *const c_void,
        beta: *const u64,
        log_arity: usize,
        beta_power0: u64,
        beta_power1: u64,
        g_inv: u64,
        ext_w: u64,
    ) -> i32;
    fn multi_stark_cuda_lookup_graph_lde(
        device_id: i32,
        output_handle: *mut *mut c_void,
        total: *mut u64,
        nodes: *const c_void,
        node_count: usize,
        slot_count: usize,
        lookups: *const c_void,
        lookup_count: usize,
        lookup_args: *const u32,
        lookup_arg_count: usize,
        preprocessed_handle: *const c_void,
        main_handle: *const c_void,
        group_size: usize,
        beta: *const u64,
        gamma: *const u64,
        ext_w: u64,
        added_bits: usize,
        inverse_twiddles: *const u64,
        shift_powers: *const u64,
        forward_twiddles: *const u64,
        height_inverse: u64,
    ) -> i32;
    fn multi_stark_cuda_lookup_lde(
        device_id: i32,
        output_handle: *mut *mut c_void,
        total: *mut u64,
        multiplicities: *const u64,
        args: *const u64,
        arg_offsets: *const usize,
        height: usize,
        num_lookups: usize,
        args_width: usize,
        group_size: usize,
        beta: *const u64,
        gamma: *const u64,
        ext_w: u64,
        added_bits: usize,
        inverse_twiddles: *const u64,
        shift_powers: *const u64,
        forward_twiddles: *const u64,
        height_inverse: u64,
    ) -> i32;
    fn multi_stark_cuda_lookup_lde_begin_partitioned(
        device_id: i32,
        pending_handle: *mut *mut c_void,
        arg_offsets: *const usize,
        height: usize,
        num_lookups: usize,
        args_width: usize,
        group_size: usize,
        added_bits: usize,
    ) -> i32;
    fn multi_stark_cuda_lookup_lde_gpu_rows_partitioned(
        device_id: i32,
        pending_handle: *mut c_void,
        multiplicities: *const u64,
        args: *const u64,
        row_start: usize,
        rows: usize,
        beta: *const u64,
        gamma: *const u64,
        ext_w: u64,
    ) -> i32;
    fn multi_stark_cuda_lookup_lde_cpu_rows_partitioned(
        device_id: i32,
        pending_handle: *mut c_void,
        deltas: *const u64,
        row_start: usize,
        rows: usize,
    ) -> i32;
    fn multi_stark_cuda_lookup_lde_finish_partitioned(
        device_id: i32,
        pending_handle: *mut c_void,
        output_handle: *mut *mut c_void,
        total: *mut u64,
        inverse_twiddles: *const u64,
        shift_powers: *const u64,
        forward_twiddles: *const u64,
        height_inverse: u64,
    ) -> i32;
    fn multi_stark_cuda_lookup_lde_cancel_partitioned(
        device_id: i32,
        pending_handle: *mut c_void,
    ) -> i32;
    fn multi_stark_cuda_reduced_create(
        device_id: i32,
        handle: *mut *mut c_void,
        height: usize,
    ) -> i32;
    fn multi_stark_cuda_reduced_add_host(
        device_id: i32,
        handle: *mut c_void,
        values: *const u64,
        height: usize,
    ) -> i32;
    #[cfg(test)]
    fn multi_stark_cuda_reduced_add(
        device_id: i32,
        reduced: *mut c_void,
        lde: *const c_void,
        height: usize,
        inv: *const u64,
        alpha: *const u64,
        reduced_y: *const u64,
        offset: *const u64,
        ext_w: u64,
    ) -> i32;
    #[cfg(test)]
    fn multi_stark_cuda_reduced_copy(
        device_id: i32,
        handle: *const c_void,
        output: *mut u64,
    ) -> i32;
    fn multi_stark_cuda_reduced_destroy(device_id: i32, handle: *mut c_void) -> i32;

    fn multi_stark_cuda_error_string(status: i32) -> *const c_char;

    fn multi_stark_cuda_blake3_hash_rows(
        device_id: i32,
        digests: *mut u8,
        messages: *const u8,
        message_bytes: usize,
        message_count: usize,
    ) -> i32;

    #[cfg(test)]
    fn multi_stark_cuda_blake3_merkle_root(
        device_id: i32,
        root: *mut u8,
        rows: *const u8,
        row_bytes: usize,
        row_count: usize,
    ) -> i32;

    #[cfg(test)]
    fn multi_stark_cuda_merkle_create(
        device_id: i32,
        handle: *mut *mut c_void,
        root: *mut u8,
        rows: *const u8,
        row_bytes: usize,
        row_count: usize,
    ) -> i32;

    #[cfg(test)]
    fn multi_stark_cuda_merkle_open(
        device_id: i32,
        handle: *const c_void,
        index: usize,
        row: *mut u8,
        siblings: *mut u8,
    ) -> i32;

    #[cfg(test)]
    fn multi_stark_cuda_merkle_destroy(device_id: i32, handle: *mut c_void) -> i32;

    #[cfg(test)]
    fn multi_stark_cuda_mixed_merkle_create(
        device_id: i32,
        handle: *mut *mut c_void,
        root: *mut u8,
        level_rows: *const *const u8,
        level_row_bytes: *const usize,
        level_heights: *const usize,
        level_count: usize,
    ) -> i32;

    fn multi_stark_cuda_mixed_merkle_open(
        device_id: i32,
        handle: *const c_void,
        index: usize,
        siblings: *mut u8,
    ) -> i32;
    fn multi_stark_cuda_mixed_merkle_open_batch(
        device_id: i32,
        handle: *const c_void,
        indices: *const u64,
        query_count: usize,
        siblings: *mut u8,
    ) -> i32;

    fn multi_stark_cuda_mixed_merkle_destroy(device_id: i32, handle: *mut c_void) -> i32;

    fn multi_stark_cuda_mixed_merkle_create_from_ldes(
        device_id: i32,
        handle: *mut *mut c_void,
        root: *mut u8,
        lde_handles: *const *const c_void,
        lde_count: usize,
    ) -> i32;
    fn multi_stark_cuda_mixed_merkle_create_hybrid(
        device_id: i32,
        handle: *mut *mut c_void,
        root: *mut u8,
        lde_handles: *const *const c_void,
        host_values: *const *const u64,
        widths: *const usize,
        heights: *const usize,
        matrix_count: usize,
        host_digest_groups: *const *const u8,
        host_digest_heights: *const usize,
        host_digest_group_count: usize,
    ) -> i32;
    fn multi_stark_cuda_hash_hybrid_height_group(
        device_id: i32,
        host_digests: *mut u8,
        lde_handles: *const *const c_void,
        host_values: *const *const u64,
        widths: *const usize,
        heights: *const usize,
        matrix_count: usize,
        height: usize,
    ) -> i32;
    fn multi_stark_cuda_fri_merkle_create(
        device_id: i32,
        handle: *mut *mut c_void,
        root: *mut u8,
        codeword: *const c_void,
        arity: usize,
    ) -> i32;

    #[cfg(test)]
    fn multi_stark_cuda_mixed_merkle_create_from_host_matrices(
        device_id: i32,
        handle: *mut *mut c_void,
        root: *mut u8,
        matrix_values: *const *const u64,
        heights: *const usize,
        widths: *const usize,
        matrix_count: usize,
    ) -> i32;

    #[cfg(test)]
    fn multi_stark_cuda_generate_coset_selectors(
        device_id: i32,
        output: *mut u64,
        quotient_size: usize,
        next_step: usize,
        coset_shift: u64,
        coset_generator: u64,
        trace_last: u64,
        vanishing_start: u64,
        vanishing_step: u64,
    ) -> i32;
}

#[cfg(test)]
mod tests {
    use super::*;
    use p3_dft::Radix2DitParallel;
    use p3_matrix::bitrev::BitReversibleMatrix;
    use p3_symmetric::CryptographicHasher;
    use rand::{RngExt, SeedableRng, rngs::SmallRng};

    #[test]
    fn device_coset_selectors_match_cpu() {
        use p3_commit::PolynomialSpace;
        use p3_field::coset::TwoAdicMultiplicativeCoset;

        for (trace_log, quotient_log) in [(4, 5), (8, 11), (12, 16), (20, 24)] {
            let trace =
                TwoAdicMultiplicativeCoset::<Goldilocks>::new(Goldilocks::ONE, trace_log).unwrap();
            let quotient =
                TwoAdicMultiplicativeCoset::<Goldilocks>::new(Goldilocks::GENERATOR, quotient_log)
                    .unwrap();
            let expected = trace.selectors_on_coset(quotient);
            let quotient_size = quotient.size();
            let next_step = quotient_size / trace.size();
            let mut actual = vec![Goldilocks::ZERO; 4 * quotient_size];
            let status = unsafe {
                multi_stark_cuda_generate_coset_selectors(
                    0,
                    actual.as_mut_ptr().cast(),
                    quotient_size,
                    next_step,
                    raw_u64(quotient.shift()),
                    raw_u64(quotient.subgroup_generator()),
                    raw_u64(trace.subgroup_generator().inverse()),
                    raw_u64(quotient.shift().exp_power_of_2(trace_log)),
                    raw_u64(Goldilocks::two_adic_generator(quotient_log - trace_log)),
                )
            };
            check_cuda(status, "coset selector generation");
            let mut flattened = expected.is_first_row;
            flattened.extend(expected.is_last_row);
            flattened.extend(expected.is_transition);
            flattened.extend(expected.inv_vanishing);
            assert_eq!(actual, flattened);
        }
    }

    #[test]
    fn host_uploaded_lde_is_canonicalized() {
        const MODULUS: u64 = 0xffff_ffff_0000_0001;
        // SAFETY: Goldilocks is repr(transparent) over u64 and accepts every
        // bit pattern; this deliberately constructs lazy representatives.
        let values = [
            unsafe { core::mem::transmute::<u64, Goldilocks>(MODULUS + 5) },
            unsafe { core::mem::transmute::<u64, Goldilocks>(MODULUS + 9) },
        ];
        let lde = CudaLde::from_row_major_matrix(0, &RowMajorMatrix::new(values.to_vec(), 1));
        assert_eq!(
            lde.to_row_major_matrix().values,
            [Goldilocks::from_u64(5), Goldilocks::from_u64(9)]
        );
    }

    #[test]
    fn resident_constraint_graph_matches_cpu() {
        use crate::expr::{CircuitSpec, ColRef, Expr, RowOffset, Source};
        use crate::graph::{ExtensionParams, compile};
        let x = Expr::Var(ColRef {
            source: Source::Main,
            offset: RowOffset::Current,
            index: 0,
        });
        let next = Expr::Var(ColRef {
            source: Source::Main,
            offset: RowOffset::Next,
            index: 0,
        });
        let spec = CircuitSpec {
            main_width: 1,
            preprocessed_width: 0,
            stage2_width: 1,
            num_publics: 1,
            constraints: vec![x.clone() * x + next - Expr::Public(0)],
            ext_constraints: vec![],
            lookups: vec![],
        };
        let graph = compile(
            &spec,
            &ExtensionParams {
                degree: 2,
                w: Goldilocks::from_u64(7),
                karatsuba: true,
            },
        )
        .unwrap();
        let height = 8;
        let logical: Vec<_> = (0..height).map(|i| Goldilocks::from_usize(i + 2)).collect();
        let bitrev = RowMajorMatrix::new_col(logical.clone())
            .bit_reverse_rows()
            .to_row_major_matrix();
        let main = CudaLde::from_row_major_matrix(0, &bitrev);
        let stage2 = CudaLde::from_row_major_matrix(
            0,
            &RowMajorMatrix::new_col(vec![Goldilocks::ZERO; height]),
        );
        let public = Goldilocks::from_u64(5);
        let selectors = vec![Goldilocks::ZERO; 3 * height];
        let got = constraint_graph_roots(
            &graph,
            None,
            &main,
            &stage2,
            &[public],
            &selectors,
            height,
            2,
        );
        for row in 0..height {
            assert_eq!(
                got.values[row],
                logical[row] * logical[row] + logical[(row + 2) % height] - public
            );
        }
    }

    #[test]
    fn resident_interpolation_matches_cpu() {
        use p3_field::coset::TwoAdicMultiplicativeCoset;
        use p3_field::{
            BasedVectorSpace, batch_multiplicative_inverse, extension::BinomialExtensionField,
        };
        use p3_matrix::interpolation::{Interpolate, compute_adjusted_weights};
        use p3_util::reverse_slice_index_bits;
        type Ext = BinomialExtensionField<Goldilocks, 2>;
        let height = 256;
        let storage_height = 512;
        let width = 1;
        let mut rng = SmallRng::seed_from_u64(991);
        let logical = RowMajorMatrix::new(
            (0..storage_height * width)
                .map(|_| rng.random::<Goldilocks>())
                .collect(),
            width,
        );
        let bitrev = logical.clone().bit_reverse_rows().to_row_major_matrix();
        let lde = CudaLde::from_row_major_matrix(0, &bitrev);
        let mut coset: Vec<_> = TwoAdicMultiplicativeCoset::new(Goldilocks::GENERATOR, 9)
            .unwrap()
            .iter()
            .collect();
        reverse_slice_index_bits(&mut coset);
        let point = Ext::from_basis_coefficients_slice(&[
            Goldilocks::from_u64(5366541241408596314),
            Goldilocks::from_u64(11432508775637878761),
        ])
        .unwrap();
        let inv =
            batch_multiplicative_inverse(&coset.iter().map(|&x| point - x).collect::<Vec<_>>());
        let adjusted = compute_adjusted_weights(point, &inv[..height]);
        let expected = bitrev
            .split_rows(height)
            .0
            .interpolate_coset_with_precomputation(Goldilocks::GENERATOR, point, &adjusted);
        let inv2: Vec<[Goldilocks; 2]> = inv[..height]
            .iter()
            .map(|x| x.as_basis_coefficients_slice().try_into().unwrap())
            .collect();
        let shift_pow = Goldilocks::GENERATOR.exp_power_of_2(8);
        let scale = (point.exp_power_of_2(8) - shift_pow)
            * (Goldilocks::from_usize(height) * shift_pow).inverse();
        let got = lde_interpolate_ext2(
            &lde,
            height,
            &inv2,
            &coset[..height],
            scale.as_basis_coefficients_slice().try_into().unwrap(),
            Goldilocks::from_u64(7),
        );
        for (got, want) in got.iter().zip(expected) {
            assert_eq!(got.as_slice(), want.as_basis_coefficients_slice());
        }
    }

    #[test]
    fn resident_reduced_opening_matches_cpu() {
        use p3_field::coset::TwoAdicMultiplicativeCoset;
        use p3_field::{
            BasedVectorSpace, batch_multiplicative_inverse, extension::BinomialExtensionField,
        };
        use p3_util::reverse_slice_index_bits;
        type Ext = BinomialExtensionField<Goldilocks, 2>;
        let height = 256;
        let width = 5;
        let logical = RowMajorMatrix::new(
            (0..height * width)
                .map(|i| Goldilocks::from_usize(i * 17 + 3))
                .collect(),
            width,
        );
        let bitrev = logical.bit_reverse_rows().to_row_major_matrix();
        let lde = CudaLde::from_row_major_matrix(0, &bitrev);
        let mut coset: Vec<_> = TwoAdicMultiplicativeCoset::new(Goldilocks::GENERATOR, 8)
            .unwrap()
            .iter()
            .collect();
        reverse_slice_index_bits(&mut coset);
        let point = Ext::from_basis_coefficients_slice(&[
            Goldilocks::from_u64(31),
            Goldilocks::from_u64(37),
        ])
        .unwrap();
        let inv =
            batch_multiplicative_inverse(&coset.iter().map(|&x| point - x).collect::<Vec<_>>());
        let inv2: Vec<[Goldilocks; 2]> = inv
            .iter()
            .map(|x| x.as_basis_coefficients_slice().try_into().unwrap())
            .collect();
        let alpha = Ext::from_basis_coefficients_slice(&[
            Goldilocks::from_u64(41),
            Goldilocks::from_u64(43),
        ])
        .unwrap();
        let ap: Vec<_> = alpha.powers().take(width).collect();
        let ap2: Vec<[Goldilocks; 2]> = ap
            .iter()
            .map(|x| x.as_basis_coefficients_slice().try_into().unwrap())
            .collect();
        let ys: Vec<_> = (0..width)
            .map(|c| point + Goldilocks::from_usize(c))
            .collect();
        let reduced_y: Ext = ap.iter().zip(&ys).map(|(&a, &y)| a * y).sum();
        let offset = alpha.exp_u64(7);
        let mut expected = vec![Ext::ZERO; height];
        for row in 0..height {
            let compressed: Ext = (0..width)
                .map(|c| ap[c] * bitrev.values[row * width + c])
                .sum();
            expected[row] = offset * (reduced_y - compressed) * inv[row];
        }
        let mut got = CudaReducedOpening::new(0, height);
        got.add(
            &lde,
            &inv2,
            &ap2,
            reduced_y.as_basis_coefficients_slice().try_into().unwrap(),
            offset.as_basis_coefficients_slice().try_into().unwrap(),
            Goldilocks::from_u64(7),
        );
        let host_values: Vec<_> = (0..height)
            .map(|row| {
                [
                    Goldilocks::from_usize(row + 11),
                    Goldilocks::from_usize(3 * row + 5),
                ]
            })
            .collect();
        got.add_host(&host_values);
        for (expected, host) in expected.iter_mut().zip(&host_values) {
            *expected += Ext::from_basis_coefficients_slice(host).unwrap();
        }
        for (got, want) in got.to_host().iter().zip(expected) {
            assert_eq!(got.as_slice(), want.as_basis_coefficients_slice());
        }
    }

    #[test]
    fn goldilocks_field_kernels_match_cpu() {
        let mut rng = SmallRng::seed_from_u64(0xc0da);
        let mut left = vec![Goldilocks::ZERO, Goldilocks::ONE];
        let mut right = vec![Goldilocks::ONE, Goldilocks::ZERO];
        left.extend((0..4096).map(|_| rng.random::<Goldilocks>()));
        right.extend((0..4096).map(|_| rng.random::<Goldilocks>()));

        let mut sums = Goldilocks::zero_vec(left.len());
        let mut differences = Goldilocks::zero_vec(left.len());
        let mut products = Goldilocks::zero_vec(left.len());
        let mut inverses = Goldilocks::zero_vec(left.len());
        // SAFETY: every input/output allocation contains `left.len()` valid
        // u64-compatible Goldilocks elements and the call is synchronous.
        let status = unsafe {
            multi_stark_cuda_goldilocks_ops(
                0,
                sums.as_mut_ptr().cast(),
                differences.as_mut_ptr().cast(),
                products.as_mut_ptr().cast(),
                inverses.as_mut_ptr().cast(),
                left.as_ptr().cast(),
                right.as_ptr().cast(),
                left.len(),
            )
        };
        check_cuda(status, "Goldilocks arithmetic contract");

        for i in 0..left.len() {
            assert_eq!(sums[i], left[i] + right[i], "sum {i}");
            assert_eq!(differences[i], left[i] - right[i], "difference {i}");
            assert_eq!(products[i], left[i] * right[i], "product {i}");
            let expected_inverse = left[i].try_inverse().unwrap_or(Goldilocks::ZERO);
            assert_eq!(inverses[i], expected_inverse, "inverse {i}");
            assert!(sums[i].as_canonical_u64() < Goldilocks::ORDER_U64);
            assert!(products[i].as_canonical_u64() < Goldilocks::ORDER_U64);
        }
    }

    #[test]
    fn blake3_rows_match_cpu_across_chunk_boundaries() {
        use p3_blake3::Blake3;

        for message_bytes in [1usize, 63, 64, 65, 1023, 1024, 1025, 4264, 7400] {
            let message_count = 17;
            let messages: Vec<u8> = (0..message_bytes * message_count)
                .map(|index| (index as u64).wrapping_mul(0x9e37_79b9).to_le_bytes()[0])
                .collect();
            let mut digests = vec![0u8; 32 * message_count];
            // SAFETY: the input contains `message_count` fixed-size messages,
            // the output has one 32-byte digest per message, and the call is
            // synchronous.
            let status = unsafe {
                multi_stark_cuda_blake3_hash_rows(
                    0,
                    digests.as_mut_ptr(),
                    messages.as_ptr(),
                    message_bytes,
                    message_count,
                )
            };
            check_cuda(status, "BLAKE3 row hashing contract");

            for (index, message) in messages.chunks_exact(message_bytes).enumerate() {
                let expected: [u8; 32] = Blake3.hash_iter(message.iter().copied());
                assert_eq!(
                    &digests[index * 32..(index + 1) * 32],
                    &expected,
                    "message_bytes={message_bytes}, index={index}"
                );
            }
        }
    }

    #[test]
    fn blake3_merkle_root_matches_cpu() {
        use p3_blake3::Blake3;

        for (row_bytes, row_count) in [(16usize, 1usize), (64, 8), (4264, 1024)] {
            let rows: Vec<u8> = (0..row_bytes * row_count)
                .map(|index| (index as u64).wrapping_mul(0x517c_c1b7).to_le_bytes()[0])
                .collect();
            let mut layer: Vec<[u8; 32]> = rows
                .chunks_exact(row_bytes)
                .map(|row| Blake3.hash_iter(row.iter().copied()))
                .collect();
            while layer.len() > 1 {
                layer = layer
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|children| {
                        Blake3.hash_iter(children[0].iter().chain(&children[1]).copied())
                    })
                    .collect();
            }
            let expected_root = layer[0];
            assert_eq!(blake3_merkle_root(0, &rows, row_bytes), expected_root);

            let tree = CudaMerkleTree::new(0, &rows, row_bytes);
            assert_eq!(tree.root(), expected_root);
            let mut index = row_count / 2;
            let (opened_row, siblings) = tree.open(index);
            assert_eq!(opened_row, rows[index * row_bytes..(index + 1) * row_bytes]);
            let mut digest: [u8; 32] = Blake3.hash_iter(opened_row);
            for sibling in siblings {
                digest = if index & 1 == 0 {
                    Blake3.hash_iter(digest.iter().chain(&sibling).copied())
                } else {
                    Blake3.hash_iter(sibling.iter().chain(&digest).copied())
                };
                index >>= 1;
            }
            assert_eq!(digest, expected_root);
        }
    }

    #[test]
    fn mixed_height_merkle_matches_cpu_layout() {
        use p3_blake3::Blake3;

        let level_8: Vec<u8> = (0usize..8 * 16)
            .map(|index| index.to_le_bytes()[0])
            .collect();
        let level_4: Vec<u8> = (0usize..4 * 24)
            .map(|index| index.wrapping_mul(17).to_le_bytes()[0])
            .collect();
        let level_1: Vec<u8> = (0usize..8)
            .map(|index| index.wrapping_mul(29).to_le_bytes()[0])
            .collect();
        let levels = [
            (level_8.as_slice(), 16usize, 8usize),
            (level_4.as_slice(), 24, 4),
            (level_1.as_slice(), 8, 1),
        ];

        let mut layer: Vec<[u8; 32]> = level_8
            .as_chunks::<16>()
            .0
            .iter()
            .map(|row| Blake3.hash_iter(row.iter().copied()))
            .collect();
        let mut digest_layers = vec![layer.clone()];
        while layer.len() > 1 {
            layer = layer
                .as_chunks::<2>()
                .0
                .iter()
                .map(|children| Blake3.hash_iter(children.iter().flatten().copied()))
                .collect();
            let injected = match layer.len() {
                4 => Some((level_4.as_slice(), 24)),
                1 => Some((level_1.as_slice(), 8)),
                _ => None,
            };
            if let Some((rows, row_bytes)) = injected {
                let row_digests = rows
                    .chunks_exact(row_bytes)
                    .map(|row| Blake3.hash_iter(row.iter().copied()));
                layer = layer
                    .into_iter()
                    .zip(row_digests)
                    .map(|(digest, row_digest)| {
                        Blake3.hash_iter(digest.iter().chain(&row_digest).copied())
                    })
                    .collect();
            }
            digest_layers.push(layer.clone());
        }

        let tree = CudaMixedMerkleTree::new(0, &levels);
        assert_eq!(tree.root(), layer[0]);
        let mut index = 5usize;
        for (actual, expected_layer) in tree.open_siblings(index).into_iter().zip(&digest_layers) {
            assert_eq!(actual, expected_layer[index ^ 1]);
            index >>= 1;
        }
    }

    #[test]
    fn batched_dft_matches_cpu() {
        let mut rng = SmallRng::seed_from_u64(0xd17);
        let cpu = Radix2DitParallel::<Goldilocks>::default();
        let gpu = CudaDft::default();
        for log_height in [0usize, 1, 2, 5, 8, 12, 14] {
            for width in [1usize, 2, 7, 31] {
                let height = 1 << log_height;
                let matrix =
                    RowMajorMatrix::new((0..height * width).map(|_| rng.random()).collect(), width);
                let expected = cpu.dft_batch(matrix.clone()).to_row_major_matrix();
                let actual = gpu.dft_batch(matrix).to_row_major_matrix();
                assert_eq!(actual, expected, "height=2^{log_height}, width={width}");
            }
        }
    }

    #[test]
    fn coset_lde_matches_cpu_including_storage_layout() {
        let mut rng = SmallRng::seed_from_u64(0x1de);
        let cpu = Radix2DitParallel::<Goldilocks>::default();
        let gpu = CudaDft::default();
        for log_height in [0usize, 1, 2, 5, 8, 12, 14] {
            for added_bits in [0usize, 1, 2, 3] {
                for width in [1usize, 2, 7] {
                    let height = 1 << log_height;
                    let matrix = RowMajorMatrix::new(
                        (0..height * width).map(|_| rng.random()).collect(),
                        width,
                    );
                    let expected = cpu
                        .coset_lde_batch(matrix.clone(), added_bits, Goldilocks::GENERATOR)
                        .bit_reverse_rows()
                        .to_row_major_matrix();
                    let actual = gpu
                        .coset_lde_batch(matrix, added_bits, Goldilocks::GENERATOR)
                        .bit_reverse_rows()
                        .to_row_major_matrix();
                    assert_eq!(
                        actual, expected,
                        "height=2^{log_height}, blowup=2^{added_bits}, width={width}"
                    );
                }
            }
        }
    }

    #[test]
    fn resident_coset_lde_matches_cpu_storage() {
        let mut rng = SmallRng::seed_from_u64(0x51de);
        let cpu = Radix2DitParallel::<Goldilocks>::default();
        let gpu = CudaDft::default();
        for (log_height, width, added_bits) in [
            (10usize, 925usize, 1usize),
            (12, 17, 1),
            (12, 2, 1),
            (14, 2, 2),
            (16, 1, 1),
        ] {
            let height = 1 << log_height;
            let matrix =
                RowMajorMatrix::new((0..height * width).map(|_| rng.random()).collect(), width);
            let expected = cpu
                .coset_lde_batch(matrix.clone(), added_bits, Goldilocks::GENERATOR)
                .bit_reverse_rows()
                .to_row_major_matrix();
            let resident = gpu.coset_lde_batch_resident(&matrix, added_bits, Goldilocks::GENERATOR);
            assert_eq!(resident.height(), expected.height());
            assert_eq!(resident.width(), expected.width());
            assert_eq!(resident.to_row_major_matrix(), expected);
        }
    }

    #[test]
    fn resident_ldes_feed_mixed_merkle_without_host_round_trip() {
        let mut rng = SmallRng::seed_from_u64(0x1de_cafe);
        let gpu = CudaDft::default();
        let inputs: Vec<RowMajorMatrix<Goldilocks>> = [(4usize, 2usize), (4, 1), (2, 3)]
            .into_iter()
            .map(|(height, width)| {
                RowMajorMatrix::new((0..height * width).map(|_| rng.random()).collect(), width)
            })
            .collect();
        let ldes: Vec<_> = inputs
            .iter()
            .map(|matrix| gpu.coset_lde_batch_resident(matrix, 1, Goldilocks::GENERATOR))
            .collect();
        let host_ldes: Vec<_> = ldes.iter().map(CudaLde::to_row_major_matrix).collect();

        let mut packed_levels = Vec::new();
        for height in [8usize, 4] {
            let matrices: Vec<_> = host_ldes
                .iter()
                .filter(|matrix| matrix.height() == height)
                .collect();
            let total_width: usize = matrices.iter().map(|matrix| matrix.width()).sum();
            let mut bytes = Vec::with_capacity(height * total_width * 8);
            for row in 0..height {
                for matrix in &matrices {
                    bytes.extend(
                        matrix
                            .row(row)
                            .unwrap()
                            .into_iter()
                            .flat_map(|value| value.as_canonical_u64().to_le_bytes()),
                    );
                }
            }
            packed_levels.push((bytes, total_width * 8, height));
        }
        let level_refs: Vec<_> = packed_levels
            .iter()
            .map(|(bytes, row_bytes, height)| (bytes.as_slice(), *row_bytes, *height))
            .collect();
        let host_tree = CudaMixedMerkleTree::new(0, &level_refs);
        let direct_host_tree = CudaMixedMerkleTree::from_host_matrices(0, &host_ldes);
        let resident_tree = CudaMixedMerkleTree::from_ldes(0, &ldes);
        let hybrid_host = vec![None, None, Some(host_ldes[2].clone())];
        let hybrid_digests = mmcs::hash_cpu_height_groups(&hybrid_host);
        let hybrid_resident = [Some(&ldes[0]), Some(&ldes[1]), None];
        let hybrid_host_refs = [None, None, Some(&host_ldes[2])];
        let hybrid_tree = CudaMixedMerkleTree::from_hybrid(
            0,
            &hybrid_resident,
            &hybrid_host_refs,
            &[None, None, None],
            &hybrid_digests,
        );
        assert_eq!(resident_tree.root(), host_tree.root());
        assert_eq!(direct_host_tree.root(), host_tree.root());
        assert_eq!(hybrid_tree.root(), host_tree.root());
        assert_eq!(resident_tree.open_siblings(5), host_tree.open_siblings(5));
        assert_eq!(hybrid_tree.open_siblings(5), host_tree.open_siblings(5));
        assert_eq!(
            direct_host_tree.open_siblings(5),
            host_tree.open_siblings(5)
        );
        for (resident, host) in ldes.iter().zip(&host_ldes) {
            let rows = [0, resident.height() / 2, resident.height() - 1];
            for row in rows {
                assert_eq!(
                    resident.row(row),
                    host.row(row).unwrap().into_iter().collect::<Vec<_>>()
                );
            }
            assert_eq!(
                resident.rows(&rows),
                rows.map(|row| host.row(row).unwrap().into_iter().collect::<Vec<_>>())
            );
        }
    }

    unsafe extern "C" {
        fn multi_stark_cuda_goldilocks_ops(
            device_id: i32,
            sums: *mut u64,
            differences: *mut u64,
            products: *mut u64,
            inverses: *mut u64,
            left: *const u64,
            right: *const u64,
            len: usize,
        ) -> i32;

    }
}
#[test]
fn ext2_schoolbook_matches_p3() {
    use p3_field::BasedVectorSpace;
    use p3_field::extension::BinomialExtensionField;
    type E = BinomialExtensionField<Goldilocks, 2>;
    let a = [
        Goldilocks::from_u64(16806794362951923611),
        Goldilocks::from_u64(17407403654981854636),
    ];
    let b = [
        Goldilocks::from_u64(15157121085234511159),
        Goldilocks::from_u64(9277427398096025811),
    ];
    let mut got = a;
    let p0 = got[0] * b[0];
    let p1 = got[1] * b[1];
    got = [
        p0 + Goldilocks::from_u64(7) * p1,
        got[0] * b[1] + got[1] * b[0],
    ];
    let expected = E::from_basis_coefficients_slice(&a).unwrap()
        * E::from_basis_coefficients_slice(&b).unwrap();
    assert_eq!(got.as_slice(), expected.as_basis_coefficients_slice());
}
