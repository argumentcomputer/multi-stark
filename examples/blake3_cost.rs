//! Compare the generic nibble gadget with an isolated custom-gate prototype.
#[path = "support/compact_blake3.rs"]
mod compact;
use multi_stark::{
    plonkish::{
        CircuitBuilder,
        gadgets::{ByteGadgets, blake3},
    },
    system::{CircuitInputs, System, SystemWitness},
    types::{CommitmentParameters, FriParameters, GoldilocksBlake3Config, Val},
};
use p3_blake3::Blake3;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{Matrix, dense::RowMajorMatrix};
use p3_symmetric::CryptographicHasher;
use std::time::Instant;

type Prepared = (
    Vec<CircuitInputs<Val>>,
    Vec<RowMajorMatrix<Val>>,
    Vec<Vec<Val>>,
);
fn baseline(messages: &[[u8; 64]], digests: &[[u8; 32]]) -> Prepared {
    let mut b = CircuitBuilder::<Val>::new();
    let bytes = ByteGadgets::new(&mut b);
    let mut inputs = vec![];
    for i in 0..messages.len() {
        let input: Vec<_> = (0..64)
            .map(|j| bytes.input(&mut b, &format!("message[{i}][{j}]")))
            .collect();
        for x in &input {
            b.expose_public(x.value());
        }
        let output = blake3(&mut b, &bytes, &input);
        for x in output {
            b.expose_public(x.value());
        }
        inputs.push(input);
    }
    let circuit = b.finish();
    println!(
        "Frontend: {:?}; layout: {:?}",
        circuit.stats(),
        circuit.multi_stark_layout().unwrap()
    );
    let mut w = circuit.witness();
    for (input, msg) in inputs.iter().zip(messages) {
        for (wire, &byte) in input.iter().zip(msg) {
            w.set(wire.value(), Val::from_u8(byte)).unwrap();
        }
    }
    let a = w.generate().unwrap();
    let public: Vec<_> = messages
        .iter()
        .zip(digests)
        .flat_map(|(m, d)| m.iter().chain(d))
        .copied()
        .map(Val::from_u8)
        .collect();
    assert_eq!(a.public_values(), public);
    let compiled = circuit.lower_to_multi_stark(Val::from_u8(92)).unwrap();
    let mut defs = compiled.circuit_inputs();
    for d in &mut defs {
        d.lookup_group_size = 3;
    }
    (
        defs,
        compiled.traces(&a).unwrap(),
        compiled.claims(&public).unwrap(),
    )
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() != 2 {
        return Err("usage: blake3_cost <baseline|compact> <hash-count>".into());
    }
    let count: usize = args[1].parse()?;
    if count == 0 {
        return Err("count must be positive".into());
    }
    let messages: Vec<[u8; 64]> = (0..count)
        .map(|i| std::array::from_fn(|j| u8::try_from((i * 73 + j * 31 + 17) % 256).unwrap()))
        .collect();
    let digests: Vec<[u8; 32]> = messages
        .iter()
        .map(|m| Blake3.hash_iter(m.iter().copied()))
        .collect();
    let start = Instant::now();
    let (defs, traces, claims) = match args[0].as_str() {
        "baseline" => baseline(&messages, &digests),
        "compact" => {
            let c = compact::Compact::new(count);
            let defs = c.definitions();
            let traces = c.traces(&messages, &defs);
            let claims = c.claims(&messages, &digests);
            assert!(compact::check(&defs, &traces, &claims));
            (defs, traces, claims)
        }
        _ => return Err("unknown mode".into()),
    };
    println!(
        "Mode={} hashes={count}; build/witness/check={:?}",
        args[0],
        start.elapsed()
    );
    let mut base_cells = 0usize;
    let mut committed_cells = 0usize;
    for (i, d) in defs.iter().enumerate() {
        let prep = d.preprocessed.as_ref().unwrap();
        let height = prep.height();
        let stage2 = d.lookups.len().max(1).div_ceil(d.lookup_group_size.max(1)) * 2;
        base_cells += height * (d.main_width + prep.width());
        committed_cells += height * (d.main_width + prep.width() + stage2);
        println!(
            "trace {i}: height={height} advice={} fixed={} stage2={stage2} lookups={}",
            d.main_width,
            prep.width(),
            d.lookups.len()
        );
    }
    println!(
        "Base advice+fixed cells={base_cells}; advice+fixed+stage2 cells={committed_cells}; 4x LDE bytes excluding quotient={}",
        committed_cells * 8 * 4
    );
    let config = GoldilocksBlake3Config::new(
        CommitmentParameters {
            log_blowup: 2,
            cap_height: 0,
        },
        FriParameters {
            log_final_poly_len: 0,
            max_log_arity: 1,
            num_queries: 100,
            commit_proof_of_work_bits: 0,
            query_proof_of_work_bits: 20,
        },
    );
    let start = Instant::now();
    let (system, key) = System::new(config, defs);
    println!("Setup={:?}", start.elapsed());
    let refs: Vec<_> = claims.iter().map(Vec::as_slice).collect();
    let start = Instant::now();
    let proof =
        system.prove_multiple_claims(&key, &refs, SystemWitness::from_stage_1(traces, &system));
    println!("Prove={:?}", start.elapsed());
    let start = Instant::now();
    system
        .verify_multiple_claims(&refs, &proof)
        .map_err(|e| format!("verification: {e:?}"))?;
    println!(
        "Verify={:?}; proof_bytes={}",
        start.elapsed(),
        proof.to_bytes()?.len()
    );
    let mut wrong = claims.clone();
    *wrong.last_mut().unwrap().last_mut().unwrap() += Val::ONE;
    assert!(
        system
            .verify_multiple_claims(&wrong.iter().map(Vec::as_slice).collect::<Vec<_>>(), &proof)
            .is_err()
    );
    println!("PASS; changed public output rejected");
    Ok(())
}
