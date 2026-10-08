mod curve;
#[path = "../../../examples/support/init_claim.rs"]
mod init_claim;
mod native_curve;
mod native_field;
mod native_msm;
mod native_transcript;
mod native_verifier;
mod outer;
mod saved;

use ark_bls12_381::{Fr, G1Affine};
use ark_ec::{AffineRepr, CurveGroup};
use ark_r1cs_std::{fields::fp::FpVar, prelude::*};
use ark_relations::r1cs::{ConstraintSystem, OptimizationGoal};
use ix_terminal_circuit::{ConstraintPhase, LinearCombination, R1csBuilder, Variable};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args
        .first()
        .is_some_and(|s| s == "native-verify" || s == "stage")
    {
        if args.len() != 3 {
            return Err("usage: init-kzg-wrap <native-verify|stage> <saved-artifact-dir> <report-or-output-dir>".into());
        }
        let dir = std::path::Path::new(&args[2]);
        if args[0] == "stage" {
            std::fs::create_dir_all(dir)?;
            return saved::check(
                std::path::Path::new(&args[1]),
                &dir.join("circuit-report.json"),
                Some(dir),
            );
        }
        return saved::check(std::path::Path::new(&args[1]), dir, None);
    }
    if args.first().is_some_and(|s| s == "prove" || s == "verify") {
        if args.len() != 2 {
            return Err("usage: init-kzg-wrap <prove|verify> <staged-dir>".into());
        }
        std::thread::spawn(|| {
            loop {
                std::thread::sleep(std::time::Duration::from_secs(30));
                if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
                    eprintln!(
                        "{}",
                        status
                            .lines()
                            .filter(|l| l.starts_with("VmRSS:") || l.starts_with("VmHWM:"))
                            .collect::<Vec<_>>()
                            .join(" ")
                    );
                }
            }
        });
        return outer::prove(std::path::Path::new(&args[1]), args[0] == "verify");
    }
    let operation = args.first().cloned().unwrap_or_else(|| "add".into());
    if args.len() > 2 || args.get(1).is_some_and(|a| a != "--weight") {
        return Err("usage: init-kzg-wrap <add|double|scalar|subgroup|subgroup-fast|subgroup-fixed> [--weight]".into());
    }
    if operation.starts_with("native-") {
        println!("{}", native_curve::measure(&operation)?);
        return Ok(());
    }
    let start = Instant::now();
    let cs = ConstraintSystem::<Fr>::new_ref();
    let weight = std::env::args().any(|a| a == "--weight");
    if weight {
        cs.set_optimization_goal(OptimizationGoal::Weight);
    }
    let g = G1Affine::generator();
    let p = curve::Point::witness(cs.clone(), g)?;
    match operation.as_str() {
        "add" => {
            let q = curve::Point::witness(cs.clone(), (g * Fr::from(7)).into_affine())?;
            p.add(&q)?
                .enforce_equal(&curve::Point::constant((g * Fr::from(8)).into_affine()))?;
        }
        "double" => {
            p.double()?
                .enforce_equal(&curve::Point::constant((g * Fr::from(2)).into_affine()))?;
        }
        "subgroup" => p.enforce_subgroup()?,
        "subgroup-fast" => p.enforce_subgroup_fast()?,
        "subgroup-fixed" => p.enforce_subgroup_fixed_chain()?,
        "scalar" => {
            let scalar = Fr::from(123456789);
            let s = FpVar::new_witness(cs.clone(), || Ok(scalar))?;
            let result = p.scalar_mul(&s.to_bits_le()?)?;
            let expected = (g * scalar).into_affine();
            assert_eq!(result.x.value()?, expected.x, "scalar x");
            assert_eq!(result.y.value()?, expected.y, "scalar y");
            result.enforce_equal(&curve::Point::constant(expected))?;
        }
        _ => {
            return Err(
                "expected add, double, scalar, subgroup, subgroup-fast, or subgroup-fixed".into(),
            );
        }
    }
    if !cs.is_satisfied()? {
        return Err(format!(
            "unsatisfied curve relation: {:?}",
            cs.which_is_unsatisfied()?
        )
        .into());
    }
    cs.finalize();
    let matrices = cs.to_matrices().ok_or("missing matrices")?;
    let gates = ix_fflonk::PlonkGateProjectionV1::new();
    let mut builder = R1csBuilder::new_projection_observed(gates.observer());
    let assignment = cs.borrow().unwrap();
    let mut variables = vec![Variable::ONE];
    for &v in &assignment.instance_assignment[1..] {
        variables.push(builder.alloc_public(v)?);
    }
    for &v in &assignment.witness_assignment {
        variables.push(builder.alloc_private(v)?);
    }
    let lc = |row: &Vec<(Fr, usize)>| {
        LinearCombination::from_terms(row.iter().map(|&(v, i)| (variables[i], v)))
    };
    for ((a, b), c) in matrices.a.iter().zip(&matrices.b).zip(&matrices.c) {
        builder.enforce(ConstraintPhase::Pcs, lc(a), lc(b), lc(c));
    }
    let projection = builder.finish_projection()?;
    let census = gates.finish_for_sizing(&projection)?;
    println!(
        "{}",
        serde_json::json!({
            "operation":operation, "optimization":if weight {"weight"} else {"constraints"},
            "constraints":cs.num_constraints(),
            "variables":variables.len(), "plonk_rows":census.active_rows(),
            "seconds":start.elapsed().as_secs_f64(), "satisfied":true,
            "scope":"Constrained primitive with generic affine formulas; not a complete KZG verifier."
        })
    );
    Ok(())
}
