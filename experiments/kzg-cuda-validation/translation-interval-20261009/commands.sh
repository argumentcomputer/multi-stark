cargo test --lib --release --features kzg,parallel --no-run --message-format=json
./test-binary plonkish::foreign::interval_tests:: --nocapture
./test-binary plonkish::foreign:: --nocapture
./test-binary plonkish::foreign::interval_tests::fixed_interval_construction_benchmark --exact --ignored --nocapture
