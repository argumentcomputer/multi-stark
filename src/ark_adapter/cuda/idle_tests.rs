use super::*;
use crate::ark_adapter::{
    KzgPcs, Scalar, Srs,
    pcs::{KzgIdleMemoryRelease, KzgProverData},
    transcript::Blake3Transcript,
};
use crate::traits::Field as _;
use crate::traits::{Algebra, Pcs};
use ark_ec::CurveGroup;
use ark_serialize::CanonicalSerialize;
use p3_matrix::dense::RowMajorMatrix;

fn report(label: &str, report: &KzgIdleMemoryRelease) {
    println!(
        "KZG_IDLE_REPORT label={label} cuda_enabled={} initialized={} quiesced={} devices={}",
        report.cuda_enabled,
        report.initialized,
        report.quiesced,
        report.devices.len()
    );
    for device in &report.devices {
        println!(
            "KZG_IDLE_RELEASE label={label} device={} released_srs_point_bytes={} released_msm_workspace_bytes={}",
            device.device, device.released_srs_point_bytes, device.released_msm_workspace_bytes
        );
        for (phase, memory) in [("before", &device.before), ("after", &device.after)] {
            println!(
                "KZG_IDLE_SNAPSHOT label={label} phase={phase} device={} driver_free_bytes={} total_bytes={} current_pool_reserved_bytes={} current_pool_used_bytes={} default_pool_reserved_bytes={} default_pool_used_bytes={} current_pool_is_default={} resident_coefficient_bytes={} srs_point_bytes={} msm_workspace_bytes={}",
                device.device,
                memory.driver_free_bytes,
                memory.total_bytes,
                memory.current_pool_reserved_bytes,
                memory.current_pool_used_bytes,
                memory.default_pool_reserved_bytes,
                memory.default_pool_used_bytes,
                memory.current_pool_is_default,
                memory.resident_coefficient_bytes,
                memory.srs_point_bytes,
                memory.msm_workspace_bytes
            );
        }
    }
}

fn assert_quiesced(report: &KzgIdleMemoryRelease, coefficients: usize) {
    assert!(report.cuda_enabled && report.initialized && report.quiesced);
    assert_eq!(
        report
            .devices
            .iter()
            .map(|device| device.device)
            .collect::<Vec<_>>(),
        Devices::get().ids
    );
    for device in &report.devices {
        assert_eq!(device.after.resident_coefficient_bytes, coefficients);
        assert_eq!(device.after.srs_point_bytes, 0);
        assert_eq!(device.after.msm_workspace_bytes, 0);
        assert_eq!(
            device.released_srs_point_bytes,
            device.before.srs_point_bytes
        );
        assert_eq!(
            device.released_msm_workspace_bytes,
            device.before.msm_workspace_bytes
        );
        assert_eq!(device.before.total_bytes, device.after.total_bytes);
        assert!(device.after.driver_free_bytes <= device.after.total_bytes);
        assert!(device.after.current_pool_used_bytes <= device.after.current_pool_reserved_bytes);
        assert!(device.after.default_pool_used_bytes <= device.after.default_pool_reserved_bytes);
        assert!(device.after.current_pool_used_bytes <= device.before.current_pool_used_bytes);
        assert!(
            device.after.current_pool_reserved_bytes <= device.before.current_pool_reserved_bytes
        );
        if device.after.current_pool_is_default {
            assert_eq!(
                device.after.current_pool_used_bytes,
                device.after.default_pool_used_bytes
            );
            assert_eq!(
                device.after.current_pool_reserved_bytes,
                device.after.default_pool_reserved_bytes
            );
        }
    }
    assert!(Devices::get().busy.lock().unwrap().iter().all(|busy| !busy));
}

fn resident_owners(data: &KzgProverData) -> Vec<usize> {
    let mut owners: Vec<_> = data.matrices[0]
        .resident_columns()
        .iter()
        .filter_map(|resident| resident.as_ref().map(|resident| resident.reservation.index))
        .collect();
    owners.sort_unstable();
    owners
}

#[test]
#[ignore = "isolated CUDA idle release, pool evidence and all-device key rehydration parity"]
fn idle_release_rehydrates_coefficients_and_preserves_opening_bytes() {
    assert!(enabled(), "requires the CUDA backend");
    assert_eq!(size_of::<NativeIdleMemory>(), 7 * size_of::<u64>());
    let devices = Devices::get();
    let n = 1 << 16;
    let width = devices.ids.len() + 1;
    let srs = Arc::new(Srs::unsafe_dev_setup(n, b"idle-memory-release"));
    let owner = srs_cache::Owner::new(Arc::clone(&srs));
    let pcs = KzgPcs::new(Arc::clone(&srs), 4);
    let srs_address = pcs.srs() as *const Srs;
    let domain = pcs.natural_domain_for_degree(n);
    let values = RowMajorMatrix::new(
        (0..n)
            .flat_map(|row| {
                (0..width).map(move |column| {
                    Scalar(if column + 1 == width {
                        Fr::from(19)
                    } else {
                        Fr::from((row as u64 * 7919 + 17) ^ (column as u64 * 65537)).square()
                    })
                })
            })
            .collect(),
        width,
    );
    let (commitment, mut data) = pcs.commit(vec![(domain, values.clone())]);
    let expected_owners: Vec<_> = (0..devices.ids.len()).collect();
    assert_eq!(resident_owners(&data), expected_owners);
    assert_eq!(
        pcs.get_evaluations_on_domain(&data, 0, domain).values,
        values.values
    );
    let weak: Vec<_> = data.matrices[0]
        .resident_columns()
        .iter()
        .flatten()
        .map(Arc::downgrade)
        .collect();
    let mut checkpoint = Vec::new();
    data.write_checkpoint(&mut checkpoint).unwrap();
    let points = vec![Scalar::from_u64(23), Scalar::from_u64(29)];
    let open = |data: &KzgProverData| {
        let (opened, proof) = pcs.open(
            vec![(data, vec![points.clone()])],
            &mut Blake3Transcript::new(),
        );
        pcs.verify(
            vec![(
                commitment.clone(),
                vec![(
                    domain,
                    points.iter().copied().zip(opened[0][0].clone()).collect(),
                )],
            )],
            &proof,
            &mut Blake3Transcript::new(),
        )
        .unwrap();
        let mut bytes = Vec::new();
        proof.0.serialize_compressed(&mut bytes).unwrap();
        (opened, bytes)
    };
    let expected = open(&data);
    let warm_srs = |data: &KzgProverData| {
        let columns = data.matrices[0].cuda_columns();
        let inputs: Vec<_> = columns[..width - 1]
            .iter()
            .map(|(column, resident, _)| (*column, *resident))
            .collect();
        let cached = srs_cache::msm_columns(&owner, 0, &inputs);
        assert_eq!(
            G1Projective::normalize_batch(&cached),
            commitment.0[0][..width - 1]
        );
        for index in 0..devices.ids.len() {
            let (points, workspace) = srs_cache::resident_bytes(index);
            assert!(points > 0 && workspace > 0);
        }
    };
    warm_srs(&data);
    let live = KzgPcs::release_idle_device_memory();
    report("live-coefficients", &live);
    assert_quiesced(&live, n * 32);
    assert!(weak.iter().all(|resident| resident.upgrade().is_some()));
    assert_eq!(open(&data), expected);
    warm_srs(&data);

    data.release_device_residency();
    assert!(weak.iter().all(|resident| resident.upgrade().is_none()));
    let released = KzgPcs::release_idle_device_memory();
    report("released", &released);
    assert_quiesced(&released, 0);
    assert!(
        released
            .devices
            .iter()
            .all(|device| device.released_srs_point_bytes > 0)
    );
    data.release_device_residency();
    let repeated = KzgPcs::release_idle_device_memory();
    report("repeated", &repeated);
    assert_quiesced(&repeated, 0);
    assert!(
        repeated
            .devices
            .iter()
            .all(|device| device.released_srs_point_bytes == 0
                && device.released_msm_workspace_bytes == 0)
    );
    for (first, second) in released.devices.iter().zip(&repeated.devices) {
        assert_eq!(
            first.after.current_pool_used_bytes,
            second.after.current_pool_used_bytes
        );
        assert_eq!(
            first.after.default_pool_used_bytes,
            second.after.default_pool_used_bytes
        );
    }
    for index in 0..devices.ids.len() {
        let device = devices.acquire_at(index);
        let available = device.budget();
        assert!(available > 0 && device.prepare_budget(available));
        assert!(!device.prepare_budget(usize::MAX));
    }

    assert_eq!(resident_owners(&data), expected_owners);
    assert_eq!(pcs.srs() as *const Srs, srs_address);
    assert_eq!(
        pcs.get_evaluations_on_domain(&data, 0, domain).values,
        values.values
    );
    assert_eq!(open(&data), expected);
    warm_srs(&data);
    let mut restored_checkpoint = Vec::new();
    data.write_checkpoint(&mut restored_checkpoint).unwrap();
    assert_eq!(restored_checkpoint, checkpoint);
    println!(
        "KZG_IDLE_OPENING bytes={} blake3={} devices={}",
        expected.1.len(),
        blake3::hash(&expected.1),
        devices.ids.len()
    );
    data.release_device_residency();
    let final_idle = KzgPcs::release_idle_device_memory();
    report("final", &final_idle);
    assert_quiesced(&final_idle, 0);
}
