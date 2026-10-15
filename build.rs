//! CUDA compilation is opt-in; CPU builds need no CUDA toolchain.

use std::env;
use std::ffi::{OsStr, OsString};
use std::path::PathBuf;
use std::process::Command;

const CUDA_EXPORTS: &str = r"{
    global: multi_stark_*; cuda_available; drop_gpu_ptr_t; clone_gpu_ptr_t;
            drop_error_message; sppark_cuda_fail;
    local: *;
};";

fn main() {
    for path in ["build.rs", "cuda"] {
        println!("cargo:rerun-if-changed={path}");
    }
    for variable in [
        "NVCC",
        "NVCC_CCBIN",
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "CUDA_HOME",
        "CUDA_PATH",
        "MULTI_STARK_CUDA_ARCHS",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    assert!(
        env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux")
            && env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("x86_64"),
        "the CUDA backend requires Linux x86_64"
    );
    let include = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("cuda");
    println!("cargo:include={}", include.display());
    let root = PathBuf::from(env::var_os("DEP_SPPARK_ROOT").expect("sppark source directory"));
    for directory in ["ff", "ntt", "util"] {
        println!("cargo:rerun-if-changed={}", root.join(directory).display());
    }
    let out = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo did not set OUT_DIR"));
    let library = out.join("libmulti_stark_cuda.so");
    let exports = out.join("exports.map");
    std::fs::write(&exports, CUDA_EXPORTS).expect("write CUDA export map");
    let nvcc = nvcc_path();
    let architectures = cuda_architectures(&nvcc);
    let compiler = || cuda_command(&nvcc, &architectures);

    let objects = [
        (PathBuf::from("cuda/sppark_ntt.cu"), "sppark_ntt.o"),
        (root.join("util/all_gpus.cpp"), "sppark_all_gpus.o"),
    ]
    .map(|(source, name)| {
        let object = out.join(name);
        run(compiler()
            .arg("-c")
            .arg(format!("-I{}", root.display()))
            .args(["-Icuda", "-DFEATURE_GOLDILOCKS"])
            .arg("-o")
            .arg(&object)
            .arg(source));
        object
    });

    // C23 conversion redirects require newer libc symbols. sppark's units
    // retain _GNU_SOURCE because libstdc++'s <mutex> needs GNU pthread APIs.
    // The private C++ runtime and unwinder prevent interposition by Lean.
    run(compiler()
        .args([
            "--shared",
            "--compiler-options=-U_GNU_SOURCE,-D_DEFAULT_SOURCE,-D_POSIX_C_SOURCE=200809L",
            "--compiler-options=-static-libstdc++,-static-libgcc",
        ])
        .arg(format!(
            "-Xlinker=-z,defs,-soname,libmulti_stark_cuda.so,--version-script,{}",
            exports.display()
        ))
        .arg("-o")
        .arg(&library)
        .arg("cuda/kernels.cu")
        .args(objects));
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=dylib=multi_stark_cuda");
    println!("cargo:library={}", library.display());
}

fn cuda_command(nvcc: &OsStr, architectures: &[String]) -> Command {
    let mut command = Command::new(nvcc);
    command.args([
        "--std=c++17",
        "--cudart=static",
        "--default-stream=per-thread",
        "-O3",
        "-lineinfo",
        "--compiler-options=-fPIC",
    ]);
    // Native cubins avoid PTX newer than the installed driver's JIT.
    for architecture in architectures {
        command.arg(format!(
            "-gencode=arch=compute_{architecture},code=sm_{architecture}"
        ));
    }
    command
}

fn run(command: &mut Command) {
    let status = command
        .status()
        .unwrap_or_else(|error| panic!("{command:?}: {error}"));
    assert!(status.success(), "{command:?} failed with {status}");
}

fn nvcc_path() -> OsString {
    env::var_os("NVCC")
        .or_else(|| {
            let root = env::var_os("CUDA_HOME").or_else(|| env::var_os("CUDA_PATH"))?;
            let path = PathBuf::from(root).join("bin/nvcc");
            path.is_file().then(|| path.into_os_string())
        })
        .unwrap_or_else(|| "nvcc".into())
}

fn cuda_architectures(nvcc: &OsStr) -> Vec<String> {
    let output = command_output(nvcc, &["--list-gpu-code"]).unwrap_or_default();
    let supported = String::from_utf8_lossy(&output);
    let supported: Vec<_> = supported
        .split_whitespace()
        .filter_map(|code| code.strip_prefix("sm_"))
        .filter(|code| code.chars().all(|c| c.is_ascii_digit()))
        .collect();
    let selected = env::var("MULTI_STARK_CUDA_ARCHS")
        .ok()
        .or_else(detect_nvidia_architectures)
        .expect("no NVIDIA GPU detected; set MULTI_STARK_CUDA_ARCHS for the target GPU");
    let architectures: Vec<_> = selected
        .split(',')
        .map(str::trim)
        .filter(|arch| !arch.is_empty())
        .map(str::to_owned)
        .collect();
    assert!(
        !architectures.is_empty(),
        "MULTI_STARK_CUDA_ARCHS must not be empty"
    );
    for arch in &architectures {
        assert!(
            (2..=3).contains(&arch.len()) && arch.chars().all(|c| c.is_ascii_digit()),
            "invalid CUDA architecture {arch:?}; expected numbers such as 80,90"
        );
        assert!(
            supported.is_empty() || supported.contains(&arch.as_str()),
            "nvcc does not support CUDA architecture sm_{arch}"
        );
    }
    architectures
}

fn command_output(program: impl AsRef<OsStr>, args: &[&str]) -> Option<Vec<u8>> {
    let output = Command::new(program).args(args).output().ok()?;
    output.status.success().then_some(output.stdout)
}

fn detect_nvidia_architectures() -> Option<String> {
    let output = command_output(
        "nvidia-smi",
        &["--query-gpu=compute_cap", "--format=csv,noheader"],
    )?;
    let capabilities = String::from_utf8(output).ok()?;
    let mut architectures: Vec<_> = capabilities
        .lines()
        .map(|line| {
            line.chars()
                .filter(|c| c.is_ascii_digit())
                .collect::<String>()
        })
        .filter(|arch| !arch.is_empty())
        .collect();
    architectures.sort_unstable();
    architectures.dedup();
    (!architectures.is_empty()).then(|| architectures.join(","))
}
