use flock_prover::pcs::ligerito::{LigeritoProfile, LigeritoSecurityConfig};
use std::{fs, path::Path};

pub fn generate(out: &Path) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(out)?;
    let mut reports = Vec::new();
    for m in 36..=40 {
        for profile in [LigeritoProfile::Fast, LigeritoProfile::Slim] {
            let config = LigeritoSecurityConfig::derive_profile(m, profile)?;
            config.validate()?;
            let toml = config.to_toml_string()?;
            let decoded = LigeritoSecurityConfig::from_toml_str(&toml)?;
            decoded.validate()?;
            decoded.to_prover_verifier_configs()?;
            fs::write(out.join(format!("m{m}_{}.toml", profile.as_str())), &toml)?;
            reports.push(serde_json::json!({
                "m": m, "profile": profile.as_str(), "initial_k": config.initial_k,
                "levels": config.levels.len(),
                "queries": config.levels.iter().map(|l| l.queries).sum::<usize>(),
                "config_blake3": blake3::hash(toml.as_bytes()).to_hex().to_string(),
            }));
        }
    }
    fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "scope": "Derived and validated PCS profiles; not a complete pipeline security audit",
            "profiles": reports,
        }))?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flock_prover::pcs::ligerito::{
        embedded_security_config, prover_config_for, verifier_config_for,
    };

    #[test]
    fn narrower_profiles_preserve_component_floors() {
        for m in [26, 36, 37, 38, 39, 40] {
            for profile in [LigeritoProfile::Fast, LigeritoProfile::Slim] {
                for k in [4, 5, 6] {
                    let config =
                        LigeritoSecurityConfig::derive_profile_with_initial_k(m, profile, k)
                            .unwrap();
                    config.validate().unwrap();
                    config.to_prover_verifier_configs().unwrap();
                    assert_eq!(config.initial_k, k);
                    if k == 6 {
                        assert_eq!(
                            config.to_toml_string().unwrap(),
                            LigeritoSecurityConfig::derive_profile(m, profile)
                                .unwrap()
                                .to_toml_string()
                                .unwrap()
                        );
                    }
                    for (index, level) in config.levels.iter().enumerate() {
                        let rho = 2.0f64.powf(-(level.log_inv_rate as f64));
                        let per_query = -(rho.sqrt() + level.eta.unwrap()).log2();
                        assert!(
                            level.queries as f64 * per_query + level.grinding_bits as f64 > 128.0
                        );
                        let mut weakened = config.clone();
                        weakened.levels[index].queries -= 1;
                        weakened.levels[index].expected_eps_query_bits =
                            (weakened.levels[index].queries as f64 * per_query * 10.0).round()
                                / 10.0;
                        assert!(
                            weakened
                                .validate()
                                .unwrap_err()
                                .contains("list-decoding target")
                        );
                    }
                }
            }
        }
        for k in [0, 31, usize::MAX] {
            assert!(
                LigeritoSecurityConfig::derive_profile_with_initial_k(38, LigeritoProfile::Slim, k)
                    .is_err()
            );
        }
        assert!(
            LigeritoSecurityConfig::derive_profile_with_initial_k(38, LigeritoProfile::Slim100, 4)
                .is_err()
        );
    }

    #[test]
    fn profile_selection_and_proofs_are_isolated() {
        for case in ["sealed", "4", "5"] {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "parameters::tests::selected_profile_child",
                    "--nocapture",
                ])
                .env("FLOCK_PROFILE_TEST", case)
                .env("RAYON_NUM_THREADS", "4")
                .status()
                .unwrap();
            assert!(status.success(), "profile case {case}");
        }
    }

    #[test]
    fn selected_profile_child() {
        use flock_prover::pcs::ligerito::{embedded_initial_k, select_initial_k};
        let Ok(case) = std::env::var("FLOCK_PROFILE_TEST") else {
            return;
        };
        let profile = LigeritoProfile::Slim;
        if case == "sealed" {
            assert_eq!(embedded_initial_k(26, profile), Some(6));
            assert!(select_initial_k(26, profile, 4).is_err());
            return;
        }
        let k: usize = case.parse().unwrap();
        select_initial_k(26, profile, k).unwrap();
        assert_eq!(embedded_initial_k(26, profile), Some(k));
        assert_eq!(embedded_initial_k(26, LigeritoProfile::Fast), Some(6));
        assert!(select_initial_k(26, profile, k).is_err());
        assert!(prover_config_for(19, k, profile).is_ok());
        assert!(verifier_config_for(19, k, profile).is_ok());
        assert!(prover_config_for(19, 6, profile).is_err());
        assert!(verifier_config_for(19, 6, profile).is_err());
        let out = std::env::temp_dir().join(format!("flock-profile-{}-{k}", std::process::id()));
        crate::fri::run_inner(&out, true, false).unwrap();
        let report: serde_json::Value =
            serde_json::from_slice(&std::fs::read(out.join("report.json")).unwrap()).unwrap();
        assert_eq!(report["dense_m"], 26);
        assert_eq!(report["initial_k"], k);
        assert_eq!(report["verified"], true);
        assert_eq!(report["altered_claim_rejected"], true);
        std::fs::remove_dir_all(out).unwrap();
    }

    #[test]
    fn extended_strict_profiles_enforce_128_bit_component_floors() {
        for m in 36..=40 {
            for profile in [LigeritoProfile::Fast, LigeritoProfile::Slim] {
                let config = LigeritoSecurityConfig::from_toml_str(
                    embedded_security_config(m, profile).unwrap(),
                )
                .unwrap();
                config.validate().unwrap();
                // The legacy target field is 100; the analysis identifier
                // selects the strict 128-bit component checks independently.
                assert!(config.analysis_version.contains("query128"));
                for (index, level) in config.levels.iter().enumerate() {
                    let rho = 2.0f64.powf(-(level.log_inv_rate as f64));
                    let per_query = -(rho.sqrt() + level.eta.unwrap()).log2();
                    let delivered = level.queries as f64 * per_query + level.grinding_bits as f64;
                    assert!(delivered > 128.0, "m{m} {profile:?} level {index}");

                    // Keep the diagnostic honest: rejection must come from
                    // the enforced floor, not inconsistent rounded metadata.
                    let mut weakened = config.clone();
                    let level = &mut weakened.levels[index];
                    level.queries -= 1;
                    level.expected_eps_query_bits =
                        (level.queries as f64 * per_query * 10.0).round() / 10.0;
                    let error = weakened.validate().unwrap_err();
                    assert!(error.contains("list-decoding target"), "{error}");

                    for claim_batch in [true, false] {
                        let mut weakened = config.clone();
                        let level = &mut weakened.levels[index];
                        let (bits, name) = if claim_batch {
                            (
                                &mut level.claim_batch_grinding_bits,
                                "claim_batch_grinding_bits",
                            )
                        } else {
                            (
                                &mut level.consistency_batch_grinding_bits,
                                "consistency_batch_grinding_bits",
                            )
                        };
                        assert!(*bits > 0);
                        *bits -= 1;
                        let error = weakened.validate().unwrap_err();
                        assert!(error.contains(name), "{error}");
                    }
                }
            }
        }
    }

    #[test]
    fn extended_profiles_are_registered_and_fail_closed() {
        for m in 36..=40 {
            for profile in [LigeritoProfile::Fast, LigeritoProfile::Slim] {
                let derived = LigeritoSecurityConfig::derive_profile(m, profile).unwrap();
                let embedded = embedded_security_config(m, profile).unwrap();
                assert_eq!(derived.to_toml_string().unwrap(), embedded);
                assert!(prover_config_for(m - 7, derived.initial_k, profile).is_ok());
                assert!(verifier_config_for(m - 7, derived.initial_k, profile).is_ok());
                assert!(prover_config_for(m - 7, derived.initial_k + 1, profile).is_err());
                let mut weakened = derived;
                weakened.levels[0].queries = 1;
                assert!(weakened.validate().is_err());
            }
        }
        assert!(prover_config_for(41 - 7, 6, LigeritoProfile::Slim).is_err());
    }
}
