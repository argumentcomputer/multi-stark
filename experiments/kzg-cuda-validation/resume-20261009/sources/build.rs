//! CUDA build isolation.
//!
//! This build script is deliberately a no-op unless Cargo enables the
//! `cuda` or `kzg-cuda` feature. Normal CPU builds therefore need neither nvcc nor CUDA
//! headers/libraries.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let kzg_cuda = env::var_os("CARGO_FEATURE_KZG_CUDA").is_some();
    if kzg_cuda {
        build_kzg_cuda();
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=cuda/kernels.cu");
    println!("cargo:rerun-if-changed=cuda/goldilocks.cuh");
    println!("cargo:rerun-if-changed=cuda/sppark_ntt.cu");
    println!("cargo:rerun-if-changed=cuda/ntt.cuh");
    if let Some(root) = env::var_os("DEP_SPPARK_ROOT") {
        let root = PathBuf::from(root);
        println!("cargo:rerun-if-changed={}", root.join("ntt").display());
        println!("cargo:rerun-if-changed={}", root.join("util").display());
    }
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=MULTI_STARK_CUDA_ARCHS");

    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        if kzg_cuda {
            link_cuda_runtime(&nvcc_path());
        }
        return;
    }

    let include = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("cuda");
    println!("cargo:include={}", include.display());

    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").as_deref(),
        Ok("linux"),
        "the first-party CUDA backend currently supports Linux targets only"
    );
    assert_eq!(
        env::var("CARGO_CFG_TARGET_ARCH").as_deref(),
        Ok("x86_64"),
        "the first-party CUDA backend currently supports x86_64 targets only"
    );

    let nvcc = nvcc_path();
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo did not set OUT_DIR"));
    let library = out_dir.join("libmulti_stark_cuda.a");
    let architectures = cuda_architectures(&nvcc);

    // sppark's units are compiled first, on their own, without the
    // `_GNU_SOURCE` undefinition below: its runtime includes libstdc++'s
    // `<mutex>`, whose GNU-only pthread functions that flag would hide. Their
    // objects then join the archive.
    let mut sppark_objects = Vec::new();
    {
        let root = PathBuf::from(
            env::var_os("DEP_SPPARK_ROOT").expect("sppark's build script exports DEP_SPPARK_ROOT"),
        );
        for (source, object) in [
            (PathBuf::from("cuda/sppark_ntt.cu"), "sppark_ntt.o"),
            (root.join("util/all_gpus.cpp"), "sppark_all_gpus.o"),
        ] {
            let object = out_dir.join(object);
            let mut compile = cuda_compile_command(&nvcc, &architectures);
            compile
                .arg("-c")
                .arg(format!("-I{}", root.display()))
                .arg("-Icuda")
                .arg("-DFEATURE_GOLDILOCKS")
                // The fork's runtime without exceptions or its thread pool:
                // no C++ runtime library symbols, so the archive links into
                // the Lean executable, which carries libc++ rather than
                // libstdc++.
                .arg("-DSPPARK_NO_CXX_RUNTIME")
                .arg("-o")
                .arg(&object)
                .arg(&source);
            let status = compile
                .status()
                .unwrap_or_else(|error| panic!("failed to execute {:?}: {error}", nvcc));
            assert!(
                status.success(),
                "nvcc failed on {} with status {status}",
                source.display()
            );
            sppark_objects.push(object);
        }
    }

    let mut command = cuda_compile_command(&nvcc, &architectures);
    command
        .arg("--lib")
        // Host code is linked by whatever toolchain links the final binary,
        // and Lean's bundled clang links against a sysroot older than
        // glibc 2.38. g++ predefines _GNU_SOURCE, under which glibc 2.38+
        // renames strtol and friends to their C23 variants (__isoc23_*),
        // which that sysroot lacks. The POSIX and default feature sets keep
        // everything the kernels' host code uses (clock_gettime, pthreads)
        // under the plain names.
        .arg("--compiler-options=-U_GNU_SOURCE,-D_DEFAULT_SOURCE,-D_POSIX_C_SOURCE=200809L")
        .arg("-o")
        .arg(&library)
        .arg("cuda/kernels.cu");
    command.args(&sppark_objects);

    let status = command.status().unwrap_or_else(|error| {
        panic!(
            "failed to execute {:?}: {error}; install the CUDA toolkit or set NVCC",
            nvcc
        )
    });
    assert!(status.success(), "nvcc failed with status {status}");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=multi_stark_cuda");
    link_cuda_runtime(&nvcc);
}

fn build_kzg_cuda() {
    assert_eq!(env::var("CARGO_CFG_TARGET_OS").as_deref(), Ok("linux"));
    assert_eq!(env::var("CARGO_CFG_TARGET_ARCH").as_deref(), Ok("x86_64"));
    let nvcc = nvcc_path();
    let architectures = cuda_architectures(&nvcc);
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let root = PathBuf::from(env::var_os("DEP_SPPARK_ROOT").expect("sppark include path"));
    let blst = PathBuf::from(env::var_os("DEP_BLST_C_SRC").expect("blst include path"));
    let headers = out.join("kzg-sppark");
    std::fs::create_dir_all(headers.join("util")).unwrap();
    let runtime = std::fs::read_to_string(root.join("util/gpu_t.cuh")).unwrap();
    let sync = "inline void sync() const\n    {\n        zero.sync();";
    assert_eq!(
        runtime.matches(sync).count(),
        1,
        "sppark gpu_t::sync changed"
    );
    // NTT parameter initialization synchronizes every GPU from one host thread.
    // Each synchronization must select the context owning those streams.
    std::fs::write(
        headers.join("util/gpu_t.cuh"),
        runtime.replace(
            sync,
            "inline void sync() const\n    {\n        select();\n        zero.sync();",
        ),
    )
    .unwrap();
    for directory in ["ec", "ff", "msm", "ntt", "polynomial", "util"] {
        println!("cargo:rerun-if-changed={}", root.join(directory).display());
    }
    println!("cargo:rerun-if-changed=cuda/kzg.cu");
    println!("cargo:rerun-if-changed=cuda/kzg_transfer.cuh");
    println!("cargo:rerun-if-changed=cuda/kzg_runtime.cpp");
    println!("cargo:rerun-if-changed=cuda/kzg_sppark.cuh");
    let mut objects = Vec::new();
    for (source, name) in [
        (PathBuf::from("cuda/kzg.cu"), "kzg.o"),
        (PathBuf::from("cuda/kzg_runtime.cpp"), "kzg_runtime.o"),
    ] {
        let object = out.join(name);
        let mut command = cuda_compile_command(&nvcc, &architectures);
        command.args([
            "-c",
            "-DFEATURE_BLS12_381",
            "-D__BLST_PORTABLE__",
            "-DMAX_LG_DOMAIN_SIZE=32",
            "-include",
            "cuda/kzg_sppark.cuh",
        ]);
        command.arg(format!("-I{}", headers.display()));
        command.arg(format!("-I{}", root.display()));
        command.arg(format!("-I{}", root.join("util").display()));
        command.arg(format!("-I{}", blst.display()));
        let status = command
            .arg(&source)
            .arg("-o")
            .arg(&object)
            .status()
            .expect("nvcc");
        assert!(
            status.success(),
            "KZG CUDA compilation failed: {}",
            source.display()
        );
        objects.push(object);
    }
    let status = Command::new(&nvcc)
        .arg("--lib")
        .args(&objects)
        .arg("-o")
        .arg(out.join("libmulti_stark_kzg_cuda.a"))
        .status()
        .expect("nvcc archive");
    assert!(status.success());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=multi_stark_kzg_cuda");
}

fn cuda_compile_command(nvcc: &std::ffi::OsStr, architectures: &[String]) -> Command {
    let mut command = Command::new(nvcc);
    command.args([
        "--std=c++17",
        "--cudart=static",
        "--default-stream=per-thread",
        "-O3",
        "-lineinfo",
        "--compiler-options=-fPIC",
    ]);
    // Native cubins avoid PTX versions newer than the installed driver's JIT,
    // even when both the toolkit and driver support the GPU's native ISA.
    for architecture in architectures {
        command.arg(format!(
            "-gencode=arch=compute_{architecture},code=sm_{architecture}"
        ));
    }
    command
}

fn link_cuda_runtime(nvcc: &std::ffi::OsStr) {
    // Rust staticlibs must carry cudart because foreign linkers do not inherit
    // Rust's dynamic native dependencies. Emit this once for both CUDA backends.
    println!("cargo:rustc-link-lib=static=cudart_static");
    for lib in ["dl", "rt", "pthread", "stdc++"] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    for directory in cuda_library_directories(nvcc) {
        if directory.is_dir() {
            println!("cargo:rustc-link-search=native={}", directory.display());
        }
    }
}

fn nvcc_path() -> std::ffi::OsString {
    if let Some(nvcc) = env::var_os("NVCC") {
        return nvcc;
    }
    if let Some(root) = env::var_os("CUDA_HOME").or_else(|| env::var_os("CUDA_PATH")) {
        let candidate = PathBuf::from(root).join("bin/nvcc");
        if candidate.is_file() {
            return candidate.into_os_string();
        }
    }
    "nvcc".into()
}

fn cuda_architectures(nvcc: &std::ffi::OsStr) -> Vec<String> {
    let supported = supported_architectures(nvcc);
    let configured = env::var("MULTI_STARK_CUDA_ARCHS").ok().map_or_else(
        || {
            let preferred = ["80", "86", "89", "90", "100", "120"];
            let selected = preferred
                .into_iter()
                .filter(|architecture| supported.iter().any(|item| item == architecture))
                .collect::<Vec<_>>();
            let defaults = if selected.is_empty() {
                detect_nvidia_architectures().unwrap_or_else(|| "80".to_owned())
            } else {
                selected.join(",")
            };
            (defaults, false)
        },
        |value| (value, true),
    );
    let (configured, explicitly_configured) = configured;
    let architectures: Vec<_> = configured
        .split(',')
        .map(str::trim)
        .filter(|architecture| !architecture.is_empty())
        .map(|architecture| {
            assert!(
                (2..=3).contains(&architecture.len())
                    && architecture.chars().all(|character| character.is_ascii_digit()),
                "invalid CUDA architecture {architecture:?}; expected comma-separated numbers such as 80,90"
            );
            assert!(
                !explicitly_configured
                    || supported.is_empty()
                    || supported.iter().any(|item| item == architecture),
                "nvcc does not support CUDA architecture sm_{architecture}"
            );
            architecture.to_owned()
        })
        .collect();
    assert!(
        !architectures.is_empty(),
        "MULTI_STARK_CUDA_ARCHS must contain at least one architecture"
    );
    architectures
}

fn supported_architectures(nvcc: &std::ffi::OsStr) -> Vec<String> {
    let Ok(output) = Command::new(nvcc).arg("--list-gpu-code").output() else {
        return Vec::new();
    };
    if !output.status.success() {
        return Vec::new();
    }
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .filter_map(|code| code.strip_prefix("sm_"))
        .filter(|code| code.chars().all(|character| character.is_ascii_digit()))
        .map(str::to_owned)
        .collect()
}

/// Detect the architecture of the installed GPU when building on the target
/// machine. This avoids requiring users to translate `nvidia-smi`'s `12.0`
/// compute capability into nvcc's `sm_120` spelling. Cross builds and hosts
/// without a visible GPU retain the portable sm_80 default and can still set
/// `MULTI_STARK_CUDA_ARCHS` explicitly.
fn detect_nvidia_architectures() -> Option<String> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let capabilities = String::from_utf8(output.stdout).ok()?;
    let mut architectures = capabilities
        .lines()
        .map(|capability| {
            capability
                .chars()
                .filter(|character| character.is_ascii_digit())
                .collect::<String>()
        })
        .filter(|architecture| !architecture.is_empty())
        .collect::<Vec<_>>();
    architectures.sort_unstable();
    architectures.dedup();
    (!architectures.is_empty()).then(|| architectures.join(","))
}

fn cuda_library_directories(nvcc: &std::ffi::OsStr) -> Vec<PathBuf> {
    let root = env::var_os("CUDA_HOME")
        .or_else(|| env::var_os("CUDA_PATH"))
        .map(PathBuf::from)
        .or_else(|| {
            let nvcc = Path::new(nvcc);
            nvcc.is_absolute()
                .then(|| nvcc.parent()?.parent().map(Path::to_path_buf))
                .flatten()
        })
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda"));
    vec![root.join("lib64"), root.join("targets/x86_64-linux/lib")]
}
