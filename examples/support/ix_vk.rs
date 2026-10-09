//! Ix's dense verifier-key transport, imported from ix e328da25.
//! This example reads a trusted VK; it is not an in-circuit byte parser.
use multi_stark::{
    expr::{ColRef, RowOffset, Source},
    graph::{ConstraintGraph, Node, NodeId},
    lookup::Lookup,
    p3_field::PrimeCharacteristicRing,
    system::{Circuit, System},
    types::{Commitment, CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val},
};
const NO_PREP_INDEX: u16 = u16::MAX;
/// Cursor over one byte region.
struct Seg<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Seg<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.pos.checked_add(n).ok_or("length overflow")?;
        if end > self.buf.len() {
            return Err(format!("eof: need {n} at offset {}", self.pos));
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> Result<u16, String> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32_usize(&mut self) -> Result<usize, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()) as usize)
    }
    fn node_id(&mut self) -> Result<NodeId, String> {
        Ok(NodeId(u32::from(self.u16()?)))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn done(&self, what: &str) -> Result<(), String> {
        if self.pos != self.buf.len() {
            return Err(format!(
                "{what}: consumed {} of {} bytes",
                self.pos,
                self.buf.len()
            ));
        }
        Ok(())
    }
}

fn decode_node(seg: &mut Seg<'_>) -> Result<Node<Val>, String> {
    Ok(match seg.u8()? {
        0 => Node::Const(Val::from_u16(seg.u16()?)),
        1 => Node::Const(Val::from_u64(seg.u64()?)),
        2 => Node::Public(u32::from(seg.u8()?)),
        3 => Node::IsFirstRow,
        4 => Node::IsLastRow,
        5 => Node::IsTransition,
        6 => Node::Add(seg.node_id()?, seg.node_id()?),
        7 => Node::Sub(seg.node_id()?, seg.node_id()?),
        8 => Node::Mul(seg.node_id()?, seg.node_id()?),
        9 => Node::Neg(seg.node_id()?),
        t @ 10..=15 => {
            let v = t - 10;
            let source = match v / 2 {
                0 => Source::Preprocessed,
                1 => Source::Main,
                _ => Source::Stage2,
            };
            let offset = if v % 2 == 0 {
                RowOffset::Current
            } else {
                RowOffset::Next
            };
            let index = u32::from(seg.u16()?);
            Node::Var(ColRef {
                source,
                offset,
                index,
            })
        }
        t => return Err(format!("bad node tag {t}")),
    })
}

/// Recompute per-node degree multiples in node order (children precede parents
/// in the compiled vector).
fn recompute_degrees(nodes: &[Node<Val>]) -> Vec<u32> {
    let mut degrees: Vec<u32> = Vec::with_capacity(nodes.len());
    for node in nodes {
        let d = match *node {
            Node::Const(_) | Node::Public(_) | Node::IsTransition => 0,
            Node::Var(_) | Node::IsFirstRow | Node::IsLastRow => 1,
            Node::Add(a, b) | Node::Sub(a, b) => degrees[a.0 as usize].max(degrees[b.0 as usize]),
            Node::Mul(a, b) => degrees[a.0 as usize] + degrees[b.0 as usize],
            Node::Neg(a) => degrees[a.0 as usize],
        };
        degrees.push(d);
    }
    degrees
}

fn decode_circuit(seg: &mut Seg<'_>) -> Result<Circuit<Val>, String> {
    let main_width = seg.u16()? as usize;
    let preprocessed_width = seg.u16()? as usize;
    let preprocessed_height = seg.u32_usize()?;
    let max_constraint_degree = seg.u16()? as usize;
    let lookup_group_size = seg.u8()? as usize;
    if !(1..=multi_stark::lookup::MAX_LOOKUP_GROUP).contains(&lookup_group_size) {
        return Err(format!("bad lookup group size {lookup_group_size}"));
    }
    let node_count = seg.u16()? as usize;
    let mut nodes = Vec::with_capacity(node_count);
    for _ in 0..node_count {
        nodes.push(decode_node(seg)?);
    }
    let zero_count = seg.u16()? as usize;
    let mut zeros = Vec::with_capacity(zero_count);
    for _ in 0..zero_count {
        zeros.push(seg.node_id()?);
    }
    let lookup_count = seg.u16()? as usize;
    let mut lookups = Vec::with_capacity(lookup_count);
    for _ in 0..lookup_count {
        let multiplicity = seg.node_id()?;
        let arg_count = seg.u16()? as usize;
        let mut args = Vec::with_capacity(arg_count.min(1 << 16));
        for _ in 0..arg_count {
            args.push(seg.node_id()?);
        }
        lookups.push(Lookup { multiplicity, args });
    }

    let degrees = recompute_degrees(&nodes);
    // The graph's own max degree covers only the user roots (the serialized
    // `max_constraint_degree` is the combined user + analytic-logUp value).
    let user_max_degree = zeros
        .iter()
        .map(|z| degrees[usize::try_from(z.0).expect("node id")])
        .max()
        .unwrap_or(0);
    // The lookup prefix is exactly the nodes interned while compiling the
    // lookup expressions, all of which are reachable from (and bounded by)
    // the lookup roots — children always precede parents.
    let lookup_prefix_len = lookups
        .iter()
        .flat_map(|l| std::iter::once(l.multiplicity).chain(l.args.iter().copied()))
        .map(|id| id.0 as usize + 1)
        .max()
        .unwrap_or(0);
    let graph = ConstraintGraph {
        nodes,
        degrees,
        zeros,
        lookups,
        lookup_prefix_len,
        max_constraint_degree: user_max_degree,
    };
    let num_lookups = graph.lookups.len();
    let ext_degree = <multi_stark::types::ExtVal as p3_field::BasedVectorSpace<Val>>::DIMENSION;
    Ok(Circuit {
        graph,
        main_width,
        preprocessed: None,
        preprocessed_width,
        preprocessed_height,
        num_lookups,
        stage_2_width: multi_stark::lookup::stage2_width(
            num_lookups,
            lookup_group_size,
            ext_degree,
        ),
        num_publics: multi_stark::lookup::num_publics(ext_degree),
        lookup_group_size,
        constraint_count: zeros_plus_logup(zero_count, num_lookups, lookup_group_size, ext_degree),
        max_constraint_degree,
    })
}

/// The folded constraint count: user roots + the directly-evaluated logUp
/// values (mirrors `multi_stark::lookup::logup_constraint_count`).
fn zeros_plus_logup(zero_count: usize, num_lookups: usize, group_size: usize, d: usize) -> usize {
    zero_count + multi_stark::lookup::logup_constraint_count(num_lookups, group_size, d)
}

/// Deserialize a `System<GoldilocksBlake3Config>` from Ix's dense VK encoding, requiring that
/// every byte is consumed. Also returns the config's construction parameters,
/// which the `System` itself doesn't expose.
pub(super) fn from_bytes(
    bytes: &[u8],
) -> Result<
    (
        System<GoldilocksBlake3Config>,
        CommitmentParameters,
        FriParameters,
    ),
    String,
> {
    let mut r = Seg { buf: bytes, pos: 0 };
    let commitment_parameters = CommitmentParameters {
        log_blowup: r.u16()? as usize,
        cap_height: r.u16()? as usize,
    };
    let fri_parameters = FriParameters {
        log_final_poly_len: r.u16()? as usize,
        max_log_arity: r.u16()? as usize,
        num_queries: r.u16()? as usize,
        commit_proof_of_work_bits: r.u16()? as usize,
        query_proof_of_work_bits: r.u16()? as usize,
    };
    let n_circuits = r.u16()? as usize;
    let mut circuits = Vec::with_capacity(n_circuits);
    for _ in 0..n_circuits {
        circuits.push(decode_circuit(&mut r)?);
    }
    let preprocessed_commit = match r.u8()? {
        0 => None,
        1 => {
            let n = r.u16()? as usize;
            let mut caps = Vec::with_capacity(n.min(1 << 16));
            for _ in 0..n {
                let mut d = [0u8; 32];
                d.copy_from_slice(r.take(32)?);
                caps.push(d);
            }
            Some(Commitment::from(caps))
        }
        t => return Err(format!("bad Option tag {t}")),
    };
    let mut preprocessed_indices = Vec::with_capacity(n_circuits);
    for _ in 0..n_circuits {
        let v = r.u16()?;
        preprocessed_indices.push(if v == NO_PREP_INDEX {
            None
        } else {
            Some(v as usize)
        });
    }
    r.done("vk")?;
    let system = System {
        config: GoldilocksBlake3Config::new(commitment_parameters, fri_parameters),
        circuits,
        preprocessed_commit,
        preprocessed_indices,
    };
    Ok((system, commitment_parameters, fri_parameters))
}
