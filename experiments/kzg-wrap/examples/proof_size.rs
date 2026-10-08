//! Size prediction from staged metadata; no full-domain allocation or proving.
use multi_stark::{
    ark_adapter::{KzgConfig, Scalar, Srs},
    expr::{RowOffset, Source},
    graph::Node,
    system::{CircuitInputs, System},
    traits::Algebra,
};
use p3_matrix::dense::RowMajorMatrix;
use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
    process::{Command, Stdio},
    sync::Arc,
};
#[derive(serde::Deserialize)]
struct Manifest {
    widths: Vec<usize>,
    heights: Vec<usize>,
    claims: Vec<Vec<Scalar>>,
}
fn load<T: serde::de::DeserializeOwned>(path: &Path) -> T {
    bincode::serde::decode_from_std_read(
        &mut BufReader::new(File::open(path).unwrap()),
        bincode::config::standard(),
    )
    .unwrap()
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arg = std::env::args().nth(1).ok_or("expected staged directory")?;
    let dir = Path::new(&arg);
    let manifest: Manifest = load(&dir.join("manifest.bin"));
    let max_height = *manifest.heights.iter().max().unwrap();
    let mut definitions = vec![];
    for i in 0..manifest.heights.len() {
        let mut process = Command::new("zstd")
            .args(["-q", "-d", "-c"])
            .arg(dir.join(format!("{i}.fixed.zst")))
            .stdout(Stdio::piped())
            .spawn()?;
        let mut header = [0; 16];
        process.stdout.as_mut().unwrap().read_exact(&mut header)?;
        process.kill()?;
        process.wait()?;
        let width = u64::from_le_bytes(header[..8].try_into()?) as usize;
        let height = u64::from_le_bytes(header[8..].try_into()?) as usize;
        assert_eq!(height, manifest.heights[i]);
        let mut definition: CircuitInputs<Scalar> = load(&dir.join(format!("{i}.meta")));
        assert_eq!(definition.main_width, manifest.widths[i]);
        // Heights and fixed values do not affect the compiled graph or widths.
        definition.preprocessed = Some(RowMajorMatrix::new(vec![Scalar::ZERO; width * 2], width));
        definitions.push(definition);
    }
    let (system, _) = System::new(
        KzgConfig::new(Arc::new(Srs::unsafe_dev_setup(2, b"size-only")), 2),
        definitions,
    );
    let mut points = 0;
    let mut fields = manifest.heights.len() - 1;
    let mut traces = vec![];
    for (i, c) in system.circuits.iter().enumerate() {
        let next = |source| {
            c.graph.nodes.iter().any(
                |n| matches!(n,Node::Var(col) if col.source==source && col.offset==RowOffset::Next),
            )
        };
        let columns = c.main_width + c.stage_2_width + c.quotient_degree();
        points += columns
            * if manifest.heights[i] < max_height {
                2
            } else {
                1
            };
        fields += c.main_width * (1 + usize::from(next(Source::Main)))
            + 2 * c.stage_2_width
            + c.quotient_degree()
            + c.preprocessed_width * (1 + usize::from(next(Source::Preprocessed)));
        traces.push(serde_json::json!({"height":manifest.heights[i],"main":c.main_width,"fixed":c.preprocessed_width,"lookup":c.stage_2_width,"quotient":c.quotient_degree(),"main_next":next(Source::Main),"fixed_next":next(Source::Preprocessed)}));
    }
    let distinct = manifest
        .heights
        .iter()
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let openings = distinct + 1;
    points += openings;
    let proof = 5 + 48 * points + 32 * fields;
    let pairing_points = (manifest.claims.len() - 1 - 18) / 2;
    let packet = proof + 18 * 8 + 32 + pairing_points * 48;
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"traces":traces,"g1_points":points,"scalar_fields":fields,"opening_witnesses":openings,"proof_bytes":proof,"pairing_bytes":pairing_points*48,"claim_bytes":18*8,"profile_bytes":32,"packet_bytes":packet,"status":"predicted from fixed shape; assumes distinct opening points"})
        )?
    );
    Ok(())
}
