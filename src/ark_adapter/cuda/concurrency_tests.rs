use super::*;
#[cfg(feature = "parallel")]
use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
#[cfg(feature = "parallel")]
use std::{
    process::Command,
    sync::Barrier,
    time::{Duration, Instant},
};

#[test]
fn device_leases_reject_reentrancy_and_release_on_unwind() {
    let devices = Devices {
        ids: vec![0, 1],
        busy: Mutex::new(vec![false; 2]),
        available: Condvar::new(),
        chunk_points: 1,
        resident_bytes: Mutex::new(vec![0; 2]),
        resident_limits: vec![0; 2],
    };
    for entry in 0..4 {
        let result = std::panic::catch_unwind(|| {
            let _lease = devices.acquire_at(0);
            match entry {
                0 => drop(devices.acquire()),
                1 => drop(devices.acquire_at(1)),
                2 => drop(devices.acquire_group(&[1])),
                _ => drop(quotient::acquire_columns(&[], &mut [], &[], 0)),
            }
        });
        assert!(result.is_err());
        assert_eq!(DEVICE_LEASES.get(), 0);
        assert!(devices.busy.lock().unwrap().iter().all(|busy| !busy));
    }
    let mut group = devices.acquire_group(&[0, 1]);
    assert_eq!(DEVICE_LEASES.get(), 2);
    drop(group.pop());
    assert_eq!(DEVICE_LEASES.get(), 1);
    drop(group);
    assert_eq!(DEVICE_LEASES.get(), 0);
    drop(devices.acquire());
    assert!(devices.busy.lock().unwrap().iter().all(|busy| !busy));
}

#[cfg(feature = "parallel")]
fn with_timeout(test: &str, operation: impl FnOnce()) {
    let name = format!("ark_adapter::cuda::concurrency_tests::{test}");
    const CHILD: &str = "MULTI_STARK_KZG_CONCURRENCY_CHILD";
    if std::env::var(CHILD).as_deref() == Ok(name.as_str()) {
        operation();
        return;
    }
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &name, "--ignored", "--nocapture"])
        .env(CHILD, &name)
        .env("RAYON_NUM_THREADS", "2")
        .env("MULTI_STARK_KZG_BACKEND", "cuda")
        .env("MULTI_STARK_KZG_CUDA_DEVICES", "0")
        .env("MULTI_STARK_KZG_CUDA_SRS_CACHE", "0")
        .env("MULTI_STARK_KZG_MSM_CHUNK_POINTS", "65536")
        .env("MULTI_STARK_KZG_CUDA_RESIDENT_GIB", "1")
        .env_remove("MULTI_STARK_KZG_CUDA_PROFILE")
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(60);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                assert!(status.success(), "{name} failed: {status}");
                return;
            }
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(20));
            }
            result => {
                let _ = child.kill();
                child.wait().unwrap();
                panic!("{name} did not complete within 60 seconds: {result:?}");
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
#[ignore = "requires one CUDA GPU; concurrent uncached MSMs with a completion deadline"]
fn concurrent_uncached_msm_completes() {
    with_timeout("concurrent_uncached_msm_completes", || {
        assert_eq!(rayon::current_num_threads(), 2);
        assert_eq!(Devices::get().ids.len(), 1);
        assert_eq!(Devices::get().chunk_points, 1 << 16);
        assert!(!srs_cache::enabled());
        let n = 1 << 20;
        let basis: Vec<_> = (1u64..=17)
            .map(|i| (G1Affine::generator() * Fr::from(i)).into_affine())
            .collect();
        let points: Vec<_> = (0..n).map(|i| basis[i % basis.len()]).collect();
        let scalars = vec![Fr::from(37); n];
        assert!(msm_normalization(&scalars).is_some());
        let expected = G1Projective::msm(&points, &scalars).unwrap();
        let start = Barrier::new(2);
        std::thread::scope(|scope| {
            let jobs: Vec<_> = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        start.wait();
                        assert_eq!(msm(&points, &scalars), expected);
                        assert_eq!(divide(&scalars, Fr::ZERO), scalars[..n - 1]);
                    })
                })
                .collect();
            for job in jobs {
                job.join().unwrap();
            }
        });
    });
}

#[cfg(feature = "parallel")]
#[test]
#[ignore = "requires one CUDA GPU; concurrent shared-key hydration with a completion deadline"]
fn concurrent_shared_matrix_hydration_completes() {
    use crate::ark_adapter::{Scalar, domain::Radix2Coset, pcs::KzgProverData};

    with_timeout("concurrent_shared_matrix_hydration_completes", || {
        assert_eq!(rayon::current_num_threads(), 2);
        let n = 1 << 16;
        let mut constant = vec![Fr::ZERO; n];
        constant[0] = Fr::from(37);
        let coefficients = vec![Fr::from(19); n];
        let data = KzgProverData::coefficient_fixture(
            Radix2Coset {
                log_size: 16,
                shift: Scalar(Fr::ONE),
            },
            vec![constant, coefficients.clone(), coefficients.clone()],
        );
        let matrix = &data.matrices[0];
        let columns: Vec<_> = (0..16)
            .into_par_iter()
            .map(|_| matrix.resident_columns())
            .collect();
        assert!(columns.iter().all(|view| std::ptr::eq(*view, columns[0])));
        let resident = columns[0];
        assert!(resident[0].is_none());
        let z = Fr::from(7);
        let expected = coefficients
            .iter()
            .rev()
            .fold(Fr::ZERO, |sum, c| sum * z + c);
        for column in &resident[1..] {
            assert!(column.is_some(), "small resident fixture must fit");
            assert_eq!(
                evaluate_resident(&coefficients, column.as_deref(), &[z]),
                [expected]
            );
        }
    });
}
