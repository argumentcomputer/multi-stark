//! Read saved circuit metadata without loading traces, parameters or a prover.

#[cfg(feature = "kzg")]
#[allow(dead_code, unreachable_pub)]
#[path = "support/kzg_storage.rs"]
mod storage;

#[cfg(feature = "kzg")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use multi_stark::{
        ark_adapter::{KzgCommitment, Scalar},
        expr::{RowOffset, Source},
        graph::Node,
        lookup::{MAX_LOOKUP_GROUP, logup_max_degree, stage2_width},
        system::Circuit,
    };
    use std::{io::Write, path::PathBuf};

    fn tuning(c: &Circuit<Scalar>, point_bytes: usize) -> (usize, usize, usize) {
        (1..=MAX_LOOKUP_GROUP)
            .filter_map(|group| {
                let degree = c
                    .graph
                    .max_constraint_degree
                    .max(logup_max_degree(&c.graph, group));
                let quotient = (degree.max(2) as usize - 1).next_power_of_two();
                let width = stage2_width(c.num_lookups, group, 1);
                (quotient <= 2).then_some((
                    width * (point_bytes + 64) + quotient * (point_bytes + 32),
                    quotient,
                    group,
                    width,
                ))
            })
            .min()
            .map(|(_, quotient, group, width)| (group, width, quotient))
            .expect("a lookup grouping within quotient budget two")
    }

    fn identity_degrees(c: &Circuit<Scalar>, height: u64) -> [u64; 3] {
        let public_degree = (1u64 << 28) - 2;
        let mut degrees: Vec<u64> = Vec::with_capacity(c.graph.nodes.len());
        for node in &c.graph.nodes {
            let degree = match *node {
                Node::Const(_) | Node::Public(_) => 0,
                Node::Var(col) => match col.source {
                    Source::Preprocessed => height - 1,
                    Source::Main | Source::Stage2 => public_degree,
                },
                Node::IsFirstRow | Node::IsLastRow => height - 1,
                Node::IsTransition => 1,
                Node::Add(a, b) | Node::Sub(a, b) => degrees[a.index()].max(degrees[b.index()]),
                Node::Mul(a, b) => degrees[a.index()] + degrees[b.index()],
                Node::Neg(a) => degrees[a.index()],
            };
            degrees.push(degree);
        }
        let air = c
            .graph
            .zeros
            .iter()
            .map(|id| degrees[id.index()])
            .max()
            .unwrap_or(0);
        let accumulator_degree = public_degree.max(height - 1);
        let lookup = c
            .graph
            .lookups
            .chunks(c.lookup_group_size)
            .map(|chunk| {
                let messages: Vec<_> = chunk
                    .iter()
                    .map(|lookup| {
                        lookup
                            .args
                            .iter()
                            .map(|id| degrees[id.index()])
                            .max()
                            .unwrap_or(0)
                    })
                    .collect();
                let sum: u64 = messages.iter().sum();
                let product = sum + accumulator_degree;
                let multiplicity = chunk
                    .iter()
                    .zip(messages)
                    .map(|(lookup, message)| degrees[lookup.multiplicity.index()] + sum - message)
                    .max()
                    .unwrap_or(0);
                product.max(multiplicity)
            })
            .max()
            .unwrap_or(accumulator_degree);
        let identity = air
            .max(lookup)
            .max(public_degree + c.quotient_degree() as u64 * height);
        [air, lookup, identity]
    }

    let mut args = std::env::args_os().skip(1);
    let manifest_dir = PathBuf::from(args.next().ok_or("expected manifest directory")?);
    let setup_dir = PathBuf::from(args.next().ok_or("expected setup directory")?);
    if args.next().is_some() {
        return Err("usage: kzg_profile_counts MANIFEST_DIR SETUP_DIR".into());
    }
    let manifest: storage::Manifest = storage::load(&manifest_dir.join("manifest.bin"))?;
    if manifest.heights.is_empty() || manifest.heights.len() != manifest.widths.len() {
        return Err("inconsistent manifest dimensions".into());
    }
    let max_height = *manifest.heights.iter().max().unwrap();
    let mut out = std::io::BufWriter::new(std::io::stdout().lock());
    writeln!(
        out,
        "index,height,main_width,fixed_width,lookups,lookup_group,lookup_width,quotient_width,main_next,fixed_next,fixed_points,shifted_fixed_points,group_48,width_48,quotient_48,group_96,width_96,quotient_96,air_full_degree,lookup_full_degree,identity_full_degree,cap27_air_full_degree,cap27_lookup_full_degree,cap27_identity_full_degree,manifest_claims,manifest_claim_fields"
    )?;
    for (index, (&height, &width)) in manifest.heights.iter().zip(&manifest.widths).enumerate() {
        let (c, fixed): (Circuit<Scalar>, KzgCommitment) =
            storage::load(&setup_dir.join(format!("setup-{index}.bin")))?;
        if c.main_width != width || c.preprocessed_height != height || c.preprocessed.is_some() {
            return Err("expected consistent metadata with preprocessing stored separately".into());
        }
        if !height.is_power_of_two() || c.num_publics != 4 || c.num_lookups != c.graph.lookups.len()
        {
            return Err("expected a base-field, power-of-two KZG profile".into());
        }
        if c.graph
            .nodes
            .iter()
            .any(|node| matches!(node, Node::Var(col) if col.source == Source::Stage2))
        {
            return Err("group retuning requires a graph without direct stage-2 references".into());
        }
        let fixed_points: usize = fixed.0.iter().map(Vec::len).sum();
        let shifted_fixed_points: usize = fixed.1.iter().map(Vec::len).sum();
        if fixed_points != c.preprocessed_width
            || shifted_fixed_points != if height < max_height { fixed_points } else { 0 }
        {
            return Err("unexpected fixed commitment shape".into());
        }
        let next = |source| {
            let used = c.graph.nodes.iter().any(|node| {
                matches!(node, Node::Var(col) if col.source == source && col.offset == RowOffset::Next)
            });
            usize::from(used)
        };
        let (group_48, width_48, quotient_48) = tuning(&c, 48);
        let (group_96, width_96, quotient_96) = tuning(&c, 96);
        let expected = if height == max_height {
            (group_48, width_48, quotient_48)
        } else {
            (group_96, width_96, quotient_96)
        };
        if expected != (c.lookup_group_size, c.stage_2_width, c.quotient_degree()) {
            return Err("saved profile differs from the current lookup tuning policy".into());
        }
        let mut row = vec![
            index as u64,
            height as u64,
            c.main_width as u64,
            c.preprocessed_width as u64,
            c.num_lookups as u64,
            c.lookup_group_size as u64,
            c.stage_2_width as u64,
            c.quotient_degree() as u64,
            next(Source::Main) as u64,
            next(Source::Preprocessed) as u64,
            fixed_points as u64,
            shifted_fixed_points as u64,
            group_48 as u64,
            width_48 as u64,
            quotient_48 as u64,
            group_96 as u64,
            width_96 as u64,
            quotient_96 as u64,
        ];
        row.extend(identity_degrees(&c, height as u64));
        row.extend(identity_degrees(&c, (height as u64).min(1 << 27)));
        row.push(manifest.claims.len() as u64);
        row.push(manifest.claims.iter().map(|claim| claim.len() as u64).sum());
        writeln!(
            out,
            "{}",
            row.iter().map(u64::to_string).collect::<Vec<_>>().join(",")
        )?;
    }
    out.flush()?;
    Ok(())
}

#[cfg(not(feature = "kzg"))]
fn main() {
    eprintln!("requires --features kzg");
    std::process::exit(1);
}
