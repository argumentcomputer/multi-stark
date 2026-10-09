//! Time preparation of one staged KZG trace, including canonical checkpoint encoding.

#[cfg(feature = "kzg")]
#[allow(dead_code, unreachable_pub)]
#[path = "support/kzg_storage.rs"]
mod storage;

#[cfg(feature = "kzg")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .init();
    use multi_stark::{
        ark_adapter::{Blake3Transcript, KzgConfig, Srs},
        config::{LookupCommitInput, ProofConfig, QuotientCommitInput},
        system::System,
        traits::{Algebra, EvaluationDomain, Pcs, Transcript},
    };
    use p3_matrix::Matrix;
    use std::{
        fs::File,
        io::{BufWriter, Write},
        path::PathBuf,
        sync::Arc,
        time::Instant,
    };

    let mut args = std::env::args().skip(1);
    let dir = PathBuf::from(args.next().expect("staged trace directory"));
    let index: usize = args.next().unwrap_or_else(|| "0".into()).parse()?;
    let quotient = args.next().is_some_and(|arg| {
        assert_eq!(arg, "quotient");
        true
    });
    assert!(args.next().is_none());
    println!("CPU threads: {}", std::thread::available_parallelism()?);
    let start = Instant::now();
    let definition = storage::definition(&dir, index)?;
    println!("read_fixed_seconds={:.6}", start.elapsed().as_secs_f64());
    let height = definition.preprocessed.as_ref().unwrap().height();
    let start = Instant::now();
    let config = KzgConfig::new(
        Arc::new(Srs::unsafe_dev_setup(height, b"prepare-benchmark")),
        2,
    )
    .with_streaming_lookups()
    .with_streaming_quotient();
    println!("srs_seconds={:.6}", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let (system, key) = System::new(config.clone(), [definition]);
    println!("commit_fixed_seconds={:.6}", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let mut output = BufWriter::with_capacity(1 << 20, File::create("/dev/null")?);
    let fixed = key.preprocessed_data.unwrap();
    fixed.write_checkpoint(&mut output)?;
    output.flush()?;
    println!("encode_fixed_seconds={:.6}", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let main = storage::read_matrix(&dir.join(format!("{index}.witness.zst")))?;
    println!("read_main_seconds={:.6}", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let (_, data) = config
        .pcs()
        .commit(vec![(config.pcs().natural_domain_for_degree(height), main)]);
    println!("commit_main_seconds={:.6}", start.elapsed().as_secs_f64());
    let start = Instant::now();
    data.write_checkpoint(&mut output)?;
    output.flush()?;
    println!("encode_main_seconds={:.6}", start.elapsed().as_secs_f64());
    let circuit = &system.circuits[0];
    let widths: Vec<_> = circuit.graph.lookups.iter().map(|l| l.args.len()).collect();
    let shape = multi_stark::lookup::LookupValues::shape_only(height, &widths);
    let mut challenger = Blake3Transcript::new();
    let beta = challenger.sample_challenge();
    let gamma = challenger.sample_challenge();
    let start = Instant::now();
    let (_, lookup, accumulators) = config
        .accelerated_lookup_commit(
            &[LookupCommitInput {
                circuit,
                lookup_values: &shape,
                preprocessed: Some((&fixed, 0)),
                stage_1: (&data, 0),
            }],
            beta,
            gamma,
            beta - beta,
        )
        .unwrap();
    println!("commit_lookup_seconds={:.6}", start.elapsed().as_secs_f64());
    if quotient {
        let start = Instant::now();
        let trace_domain = config.pcs().natural_domain_for_degree(height);
        let quotient_domain =
            trace_domain.create_disjoint_domain(height * circuit.quotient_degree());
        let _quotient = config
            .accelerated_quotient_commit(
                &[QuotientCommitInput {
                    circuit,
                    lookup_publics: vec![
                        beta,
                        gamma,
                        multi_stark::ark_adapter::Scalar::ZERO,
                        accumulators[0],
                    ],
                    trace_domain,
                    quotient_domain,
                    preprocessed: Some((&fixed, 0)),
                    stage_1: (&data, 0),
                    stage_2: (&lookup, 0),
                    constraint_count: circuit.constraint_count(),
                }],
                challenger.sample_challenge(),
            )
            .unwrap();
        println!(
            "commit_quotient_seconds={:.6}",
            start.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

#[cfg(not(feature = "kzg"))]
fn main() {
    eprintln!("Enable kzg or kzg-cuda to benchmark KZG preparation.");
}
