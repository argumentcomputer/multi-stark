use super::*;
use ark_ec::{AffineRepr, VariableBaseMSM};

#[test]
fn owner_identity_is_unique_and_keeps_parameters_immutable() {
    let mut source = Arc::new(Srs::unsafe_dev_setup(8, b"srs-owner"));
    let owner = Owner::new(Arc::clone(&source));
    let another = Owner::new(Arc::clone(&source));
    assert_ne!(owner.id, another.id);
    assert!(Arc::get_mut(&mut source).is_none());
    let weak = Arc::downgrade(&owner);
    let cloned = Arc::clone(&owner);
    drop(owner);
    assert!(weak.upgrade().is_some());
    drop(cloned);
    assert!(weak.upgrade().is_none());
    drop(another);
    assert!(Arc::get_mut(&mut source).is_some());
}

#[test]
fn ranges_require_the_same_owner_and_complete_coverage() {
    let source = Arc::new(Srs::unsafe_dev_setup(8, b"srs-ranges"));
    let owner = Owner::new(Arc::clone(&source));
    let other = Owner::new(source);
    let entry = Entry {
        owner: Arc::downgrade(&owner),
        id: owner.id,
        start: 2,
        count: 4,
        pointer: 0,
        bytes: 0,
        used: 0,
    };
    assert!(entry.contains(&owner, 2, 4));
    assert!(entry.contains(&owner, 3, 2));
    assert!(!entry.contains(&other, 2, 4));
    assert!(!entry.contains(&owner, 1, 2));
    assert!(!entry.contains(&owner, 5, 2));
    assert!(!entry.contains(&owner, usize::MAX, 1));
    assert!(entry.overlaps(&owner, 1, 2));
    assert!(!entry.overlaps(&owner, 6, 2));
}

fn coefficients(count: usize) -> Vec<Fr> {
    (0..count)
        .map(|i| Fr::from((i as u64).wrapping_mul(7919) + 17).square())
        .collect()
}

fn report_cache(label: &str) {
    for (index, cache) in caches().iter().enumerate() {
        let cache = cache.lock().unwrap();
        eprintln!(
            "srs_cache_state label={label} device={} hits={} uploads={} upload_bytes={} point_evictions={} point_bytes={} workspace_bytes={}",
            Devices::get().ids[index],
            cache.hits,
            cache.uploads,
            cache.upload_bytes,
            cache.evictions,
            cache.point_bytes(),
            cache
                .workspace
                .as_ref()
                .map_or(0, |workspace| workspace.bytes),
        );
    }
}

#[test]
#[ignore = "requires CUDA; checks direct prefix/subrange hits and scalar-pointer rebinding"]
fn cached_srs_prefix_hits_after_normalization() {
    let count = 1 << 16;
    let owner = Owner::new(Arc::new(Srs::unsafe_dev_setup(count, b"cached-srs-prefix")));
    let input = coefficients(count);
    let structured = vec![Fr::from(37); count];
    let resident = retain(&input).expect("small resident fixture must fit");
    let device = Devices::get().acquire_at(resident.reservation.index);
    let full = prepare(&device, &owner, 0, count);
    let normalization = msm_normalization(&structured);
    assert!(normalization.is_some());
    assert_eq!(
        invoke(&device, &full, Scalars::Host(&structured), normalization),
        G1Projective::msm(&owner.srs.g1[..count], &structured).unwrap()
    );
    let (warm_hits, warm_uploads) = {
        let cache = caches()[device.index].lock().unwrap();
        (cache.hits, cache.uploads)
    };
    for (point_offset, scalar_offset, length) in [(0, 0, 4096), (37, 113, 4099)] {
        let hit = prepare(&device, &owner, point_offset, length);
        assert_eq!(hit.points, full.points);
        assert_eq!(hit.point_offset, point_offset);
        assert_eq!(hit.workspace, full.workspace);
        assert_eq!(
            invoke(
                &device,
                &hit,
                Scalars::Resident {
                    pointer: resident.context,
                    offset: scalar_offset,
                    count: length,
                },
                None,
            ),
            G1Projective::msm(
                &owner.srs.g1[point_offset..point_offset + length],
                &input[scalar_offset..scalar_offset + length],
            )
            .unwrap()
        );
    }
    let hit = prepare(&device, &owner, 0, count);
    assert_eq!(hit.points, full.points);
    assert_eq!(
        invoke(&device, &hit, Scalars::Host(&input), None),
        G1Projective::msm(&owner.srs.g1[..count], &input).unwrap()
    );
    {
        let cache = caches()[device.index].lock().unwrap();
        assert_eq!(cache.uploads, warm_uploads);
        assert_eq!(cache.hits, warm_hits + 3);
    }
    report_cache("prefix-rebinding");
}

#[test]
#[ignore = "requires CUDA; checks cache hits, ownership, ranges, normalization and eviction"]
fn cached_srs_msm_parity_and_admission() {
    let mut source = Srs::unsafe_dev_setup(1 << 17, b"cached-srs-parity");
    source.g1[0] = G1Affine::zero();
    source.g1[1] = G1Affine::generator();
    source.g1[2] = -source.g1[1];
    source.g1[4].infinity = true;
    let owner = Owner::new(Arc::new(source));
    let input = coefficients(1 << 16);
    for (offset, length) in [(0, 257), (31, 4099), (0, 1 << 16), (17, (1 << 16) - 1)] {
        let expected =
            G1Projective::msm(&owner.srs.g1[offset..offset + length], &input[..length]).unwrap();
        let actual = msm_columns(&owner, offset, &[(&input[..length], None)])[0];
        assert_eq!(actual, expected);
        assert_eq!(
            msm_columns(&owner, offset, &[(&input[..length], None)])[0],
            expected
        );
    }
    let structured = vec![Fr::from(37); input.len()];
    let resident = retain(&structured).expect("small resident fixture must fit");
    let expected = G1Projective::msm(&owner.srs.g1[..structured.len()], &structured).unwrap();
    assert_eq!(
        msm_columns(&owner, 0, &[(&structured, Some(&resident))])[0],
        expected
    );
    let other = Owner::new(Arc::new(Srs::unsafe_dev_setup(
        1 << 17,
        b"different-srs-parity",
    )));
    let expected_other = G1Projective::msm(&other.srs.g1[..input.len()], &input).unwrap();
    assert_eq!(msm_columns(&other, 0, &[(&input, None)])[0], expected_other);
    assert_ne!(
        expected_other,
        G1Projective::msm(&owner.srs.g1[..input.len()], &input).unwrap()
    );
    let devices = Devices::get();
    for index in 0..devices.ids.len() {
        let device = devices.acquire_at(index);
        if reclaimable(index) != 0 {
            let required = device.budget() + 1;
            assert!(devices.potential_budget(index) >= required);
            assert!(device.prepare_budget(required));
            assert!(device.budget() >= required);
        }
        let mut cache = caches()[index].lock().unwrap();
        while let Some(index) = cache.oldest() {
            cache.evict(&device, index);
        }
        cache.drop_workspace(&device);
        assert_eq!(cache.bytes(), 0);
    }
    assert_eq!(
        msm_columns(&owner, 0, &[(&structured, Some(&resident))])[0],
        expected
    );
    drop(resident);
    let weak = Arc::downgrade(&owner);
    drop(owner);
    assert!(weak.upgrade().is_none());
    for index in 0..devices.ids.len() {
        let device = devices.acquire_at(index);
        assert!(device.prepare_budget(0));
        assert!(
            caches()[index]
                .lock()
                .unwrap()
                .entries
                .iter()
                .all(|entry| entry.owner.strong_count() > 0)
        );
    }
    report_cache("ownership-ranges-eviction");
}

#[test]
#[ignore = "requires exactly one selected CUDA device; checks cached opening division and normalization"]
fn cached_srs_single_device_opening_parity() {
    assert_eq!(Devices::get().ids.len(), 1);
    let owner = Owner::new(Arc::new(Srs::unsafe_dev_setup(
        1 << 16,
        b"cached-srs-opening",
    )));
    let input = coefficients(1 << 16);
    let structured = vec![Fr::from(37); input.len()];
    assert!(single_device_opening(&owner, &[], Fr::ONE).is_none());
    assert!(single_device_opening(&owner, &[Fr::ONE], Fr::ONE).is_none());
    let mut uploads = None;
    for (coefficients, z) in [
        (input.as_slice(), Fr::ZERO),
        (structured.as_slice(), Fr::ZERO),
        (input.as_slice(), Fr::from(17)),
    ] {
        let mut quotient = vec![Fr::ZERO; coefficients.len() - 1];
        let mut carry = Fr::ZERO;
        for index in (1..coefficients.len()).rev() {
            carry = coefficients[index] + z * carry;
            quotient[index - 1] = carry;
        }
        assert_eq!(
            single_device_opening(&owner, coefficients, z).unwrap(),
            G1Projective::msm(&owner.srs.g1[..quotient.len()], &quotient).unwrap()
        );
        let cache = caches()[0].lock().unwrap();
        if let Some(uploads) = uploads {
            assert_eq!(cache.uploads, uploads);
        } else {
            uploads = Some(cache.uploads);
        }
    }
    report_cache("single-device-opening");
}

#[test]
#[ignore = "isolated alternating repeated-MSM comparison; requires CUDA"]
fn cached_srs_repeated_msm_benchmark() {
    use ark_ff::PrimeField;
    use std::time::Instant;
    let log: u32 =
        std::env::var("MULTI_STARK_KZG_SRS_BENCH_LOG_N").map_or(20, |value| value.parse().unwrap());
    assert!((18..=23).contains(&log));
    let owner = Owner::new(Arc::new(Srs::unsafe_dev_setup(
        1 << log,
        b"srs-cache-benchmark",
    )));
    let first: Vec<_> = (0usize..1 << log)
        .into_par_iter()
        .map(|index| {
            let mut bytes = [0; 64];
            blake3::Hasher::new()
                .update(&(index as u64).to_le_bytes())
                .finalize_xof()
                .fill(&mut bytes);
            Fr::from_le_bytes_mod_order(&bytes)
        })
        .collect();
    let second: Vec<_> = first.iter().map(|value| -*value).collect();
    let columns = [first.as_slice(), second.as_slice()];
    let cached_columns = [(columns[0], None), (columns[1], None)];
    let reference = super::super::msm_columns(&owner.srs.g1, &columns);
    assert_eq!(msm_columns(&owner, 0, &cached_columns), reference);
    report_cache("warmup");
    for iteration in 0..5 {
        for cached in if iteration % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let start = Instant::now();
            let actual = if cached {
                msm_columns(&owner, 0, &cached_columns)
            } else {
                super::super::msm_columns(&owner.srs.g1, &columns)
            };
            let seconds = start.elapsed().as_secs_f64();
            assert_eq!(actual, reference);
            eprintln!(
                "srs_cache_sample iteration={iteration} cached={cached} log_n={log} seconds={seconds:.6}"
            );
            report_cache(&format!("sample-{iteration}-{cached}"));
        }
    }
}
