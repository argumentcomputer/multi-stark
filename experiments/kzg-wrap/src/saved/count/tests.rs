use super::*;
use multi_stark::plonkish::CircuitStats;

fn counts(
    gates: usize,
    publics: usize,
    dimensions: Vec<(usize, usize)>,
    pairing_points: usize,
) -> Counts {
    Counts {
        stats: CircuitStats {
            gates,
            publics,
            tables: dimensions.len(),
            ..CircuitStats::default()
        },
        table_dimensions: dimensions,
        degree_output_count: pairing_points.saturating_sub(2),
        msm_terms: vec![0; pairing_points],
    }
}

#[test]
fn census_reports_oversized_rows_and_exact_merged_table_boundary() {
    let mut counted = counts(7, 0, vec![(5, 1), (3, 3)], 2);
    let layout = Layout::from_counts(&counted, 8).unwrap();
    assert!(layout.layout_admissible);
    assert_eq!((layout.ordinary_rows, layout.main_height), (8, 8));
    assert_eq!(
        (layout.merged_table_rows, layout.merged_table_height),
        (8, 8)
    );
    assert_eq!(
        (
            layout.main_width,
            layout.main_fixed_width,
            layout.merged_table_fixed_width
        ),
        (3, 15, 4)
    );

    counted.stats.gates += 1;
    let layout = Layout::from_counts(&counted, 8).unwrap();
    assert!(!layout.layout_admissible);
    assert_eq!((layout.ordinary_rows, layout.main_height), (9, 16));
    assert_eq!(
        projection(&layout, &counted, false).unwrap()["status"],
        "unavailable_for_rejected_layout"
    );

    counted.stats.gates -= 1;
    counted.table_dimensions[0].0 += 1;
    let layout = Layout::from_counts(&counted, 8).unwrap();
    assert!(!layout.layout_admissible);
    assert_eq!(
        (layout.merged_table_rows, layout.merged_table_height),
        (9, 16)
    );
    assert_eq!(layout.rejection_reasons.len(), 2);

    counted.table_dimensions[0].0 -= 1;
    counted.stats.hash_calls = 1;
    let layout = Layout::from_counts(&counted, 8).unwrap();
    assert!(!layout.row_count_complete && !layout.layout_admissible);
    counted.stats.hash_calls = 0;
    counted.stats.gates = usize::MAX;
    assert!(Layout::from_counts(&counted, 8).is_err());
    assert!(Layout::from_counts(&counted, 3).is_err());
}

#[test]
fn census_codec_projects_current_public_packet_and_rejects_large_payloads() {
    let tables = vec![(65536, 1), (16, 1), (256, 3), (16, 3)];
    let counted = counts(HEIGHT_CAP - 23, 22, tables.clone(), 2);
    let layout = Layout::from_counts(&counted, HEIGHT_CAP).unwrap();
    assert_eq!(layout.main_height, HEIGHT_CAP);
    assert_eq!(layout.merged_table_height, 1 << 17);
    let packet = projection(&layout, &counted, false).unwrap();
    assert_eq!(packet["projected_proof_bytes"], 1909);
    assert_eq!(packet["projected_packet_bytes"], 2181);
    assert_eq!(packet["projected_packet_below_limit"], true);
    assert_eq!(packet["actual_outer_proof_generated"], false);
    assert_eq!(packet["opening_witnesses"], 3);
    for (points, expected_bytes, admitted) in [(19, 2997, true), (20, 3045, false)] {
        let counted = counts(1 << 20, 18 + 2 * points, tables.clone(), points);
        let layout = Layout::from_counts(&counted, HEIGHT_CAP).unwrap();
        let packet = projection(&layout, &counted, false).unwrap();
        assert_eq!(packet["projected_packet_bytes"], expected_bytes);
        assert_eq!(packet["projected_packet_below_limit"], admitted);
        assert_eq!(packet["packet_limit_exclusive"], 3000);
    }
    let mut malformed = counts(1 << 20, 22, tables, 2);
    malformed.stats.publics += 1;
    let layout = Layout::from_counts(&malformed, HEIGHT_CAP).unwrap();
    assert!(projection(&layout, &malformed, false).is_err());
}

#[test]
fn missing_runtime_checkout_is_explicit_and_nonfatal() {
    let missing = std::env::temp_dir().join(format!(
        "missing-census-checkout-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
    ));
    let provenance = source_hashes_at(&missing);
    assert_eq!(provenance["complete"], false);
    assert!(provenance["hashes_blake3"].as_object().unwrap().is_empty());
    assert!(!provenance["unavailable"].as_object().unwrap().is_empty());
    assert!(
        provenance["scope"]
            .as_str()
            .unwrap()
            .contains("not authenticated build provenance")
    );
}
