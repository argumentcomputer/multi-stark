// CUDA kernel compilation. Runs only when the `cuda` feature is enabled
// (Cargo sets CARGO_FEATURE_CUDA in that case). Otherwise the build script
// is a no-op and the crate builds as a pure-Rust library.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }

    println!("cargo:rerun-if-changed=cuda/gpu_api.cu");
    println!("cargo:rerun-if-changed=cuda/poseidon2.cu");
    println!("cargo:rerun-if-env-changed=NVCC");

    let sppark_root = std::env::var_os("DEP_SPPARK_ROOT")
        .expect("sppark dependency must publish DEP_SPPARK_ROOT (enable the `cuda` feature)");

    let mut nvcc = cc::Build::new();
    nvcc.cuda(true);

    // Blackwell (sm_120) is the primary target; ship sm_80 PTX as a forward-compatible
    // fallback for Ampere and Hopper.
    nvcc.flag("-arch=sm_120");
    nvcc.flag("-gencode").flag("arch=compute_80,code=sm_80");

    // Quiet some warnings emitted by the vendored sppark headers.
    nvcc.flag("-Xcompiler").flag("-Wno-unused-function");
    nvcc.flag("-Xcompiler").flag("-Wno-unused-parameter");

    // Hardcoded to the only field multi-stark targets.
    nvcc.define("FEATURE_GOLDILOCKS", None);
    // Skip sppark's slow-path reduction kludge for Goldilocks; the headers
    // include the fast Solinas reduction unconditionally when this is set.
    nvcc.define("GL64_NO_REDUCTION_KLUDGE", None);
    // Have sppark's RustError carry the underlying message string instead of
    // an empty placeholder.
    nvcc.define("TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE", None);

    nvcc.include(&sppark_root);
    nvcc.file("cuda/gpu_api.cu");
    nvcc.file("cuda/poseidon2.cu");

    nvcc.compile("multi_stark_cuda");
}
