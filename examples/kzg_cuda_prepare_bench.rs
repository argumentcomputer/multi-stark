//! Time staged KZG partitions, including canonical checkpoint encoding.

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
    let indices: Vec<usize> = args
        .next()
        .unwrap_or_else(|| "0".into())
        .split(',')
        .map(str::parse)
        .collect::<Result<_, _>>()?;
    assert!(!indices.is_empty());
    let quotient = args.next().is_some_and(|arg| {
        assert_eq!(arg, "quotient");
        true
    });
    assert!(args.next().is_none());
    let prefetch_gib: usize =
        std::env::var("MULTI_STARK_KZG_PREFETCH_GIB").map_or(Ok(32), |value| value.parse())?;
    let prefetch_bytes = prefetch_gib
        .checked_mul(1 << 30)
        .ok_or("prefetch budget overflow")?;
    println!("CPU threads: {}", std::thread::available_parallelism()?);
    println!("Partitions: {indices:?}; prefetch budget: {prefetch_gib} GiB");
    let start = Instant::now();
    let definitions = indices
        .iter()
        .map(|&i| storage::definition(&dir, i))
        .collect::<storage::Result<Vec<_>>>()?;
    println!("read_fixed_seconds={:.6}", start.elapsed().as_secs_f64());
    let heights: Vec<_> = definitions
        .iter()
        .map(|d| d.preprocessed.as_ref().unwrap().height())
        .collect();
    let start = Instant::now();
    let mut config = KzgConfig::new(
        Arc::new(Srs::unsafe_dev_setup(
            *heights.iter().max().unwrap(),
            b"prepare-benchmark",
        )),
        2,
    )
    .with_streaming_lookups()
    .with_partition_pipeline(prefetch_bytes);
    if indices.len() == 1 {
        config = config.with_streaming_quotient();
    }
    println!("srs_seconds={:.6}", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let (system, key) = System::new(config.clone(), definitions);
    println!("commit_fixed_seconds={:.6}", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let mut output = BufWriter::with_capacity(1 << 20, File::create("/dev/null")?);
    let fixed = key.preprocessed_data.unwrap();
    fixed.write_checkpoint(&mut output)?;
    output.flush()?;
    println!("encode_fixed_seconds={:.6}", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let main = indices
        .iter()
        .zip(&heights)
        .map(|(&i, &height)| {
            Ok((
                config.pcs().natural_domain_for_degree(height),
                storage::read_matrix(&dir.join(format!("{i}.witness.zst")))?,
            ))
        })
        .collect::<storage::Result<Vec<_>>>()?;
    println!("read_main_seconds={:.6}", start.elapsed().as_secs_f64());
    let start = Instant::now();
    let (_, data) = config.pcs().commit(main);
    println!("commit_main_seconds={:.6}", start.elapsed().as_secs_f64());
    let start = Instant::now();
    data.write_checkpoint(&mut output)?;
    output.flush()?;
    println!("encode_main_seconds={:.6}", start.elapsed().as_secs_f64());
    let shapes: Vec<_> = system
        .circuits
        .iter()
        .zip(&heights)
        .map(|(circuit, &height)| {
            let widths: Vec<_> = circuit.graph.lookups.iter().map(|l| l.args.len()).collect();
            multi_stark::lookup::LookupValues::shape_only(height, &widths)
        })
        .collect();
    let mut challenger = Blake3Transcript::new();
    let beta = challenger.sample_challenge();
    let gamma = challenger.sample_challenge();
    let start = Instant::now();
    let (_, lookup, accumulators) = config
        .accelerated_lookup_commit(
            &system
                .circuits
                .iter()
                .zip(&shapes)
                .enumerate()
                .map(|(i, (circuit, shape))| LookupCommitInput {
                    circuit,
                    lookup_values: shape,
                    preprocessed: Some((&fixed, i)),
                    stage_1: (&data, i),
                })
                .collect::<Vec<_>>(),
            beta,
            gamma,
            beta - beta,
        )
        .unwrap();
    println!("commit_lookup_seconds={:.6}", start.elapsed().as_secs_f64());
    if quotient {
        let start = Instant::now();
        let mut previous = multi_stark::ark_adapter::Scalar::ZERO;
        let inputs: Vec<_> = system
            .circuits
            .iter()
            .zip(&heights)
            .enumerate()
            .map(|(i, (circuit, &height))| {
                let trace_domain = config.pcs().natural_domain_for_degree(height);
                let quotient_domain =
                    trace_domain.create_disjoint_domain(height * circuit.quotient_degree());
                let input = QuotientCommitInput {
                    circuit,
                    lookup_publics: vec![beta, gamma, previous, accumulators[i]],
                    trace_domain,
                    quotient_domain,
                    preprocessed: Some((&fixed, i)),
                    stage_1: (&data, i),
                    stage_2: (&lookup, i),
                    constraint_count: circuit.constraint_count(),
                };
                previous = accumulators[i];
                input
            })
            .collect();
        let alpha = challenger.sample_challenge();
        let _quotient = config
            .accelerated_quotient_commit(&inputs, alpha)
            .unwrap_or_else(|| {
                let values = inputs
                    .iter()
                    .map(|input| {
                        let values = config
                            .accelerated_quotient_values(
                                input.circuit,
                                &input.lookup_publics,
                                input.trace_domain,
                                input.quotient_domain,
                                input.preprocessed,
                                input.stage_1,
                                input.stage_2,
                                alpha,
                                input.constraint_count,
                            )
                            .unwrap();
                        (
                            input.quotient_domain,
                            p3_matrix::dense::RowMajorMatrix::new_col(values),
                            input.quotient_domain.size() / input.trace_domain.size(),
                        )
                    })
                    .collect();
                config.pcs().commit_quotient(values)
            });
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
