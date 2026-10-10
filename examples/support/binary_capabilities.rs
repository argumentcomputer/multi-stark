pub(crate) fn print(binary: &str) {
    let compiled = multi_stark::BuildCapabilities::compiled();
    println!(
        concat!(
            "{{\"format\":\"multi-stark-capabilities/v1\",\"binary\":\"{}\",",
            "\"kzg\":{},\"kzg_cuda\":{},\"goldilocks_cuda\":{},\"parallel\":{}}}"
        ),
        binary, compiled.kzg, compiled.kzg_cuda, compiled.goldilocks_cuda, compiled.parallel,
    );
}
