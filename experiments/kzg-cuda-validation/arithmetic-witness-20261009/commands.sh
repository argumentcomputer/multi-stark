cargo test --lib --release --features kzg,parallel --no-run --message-format=json
./test-binary plonkish::witness::arithmetic_tests:: --nocapture
./test-binary plonkish::foreign::tests:: --nocapture
./test-binary plonkish::witness::arithmetic_tests::specialized_arithmetic_benchmark --exact --ignored --nocapture
