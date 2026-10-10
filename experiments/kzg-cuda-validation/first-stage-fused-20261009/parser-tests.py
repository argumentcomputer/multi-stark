import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("bench", Path(__file__).with_name("kzg-cuda-bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class BenchmarkEvidenceTests(unittest.TestCase):
    def test_distributed_wrapper_budget_is_scoped_to_outer_gpu_process(self):
        base = {
            "MULTI_STARK_KZG_BACKEND": "cuda",
            "MULTI_STARK_KZG_CUDA_RESIDENT_GIB": "32",
            "MULTI_STARK_KZG_CUDA_DISTRIBUTED_QUOTIENT": "0",
            "MULTI_STARK_KZG_CUDA_DISTRIBUTED_LOOKUP": "0",
            "MULTI_STARK_KZG_SETUP": "filecoin",
            "MULTI_STARK_KZG_FILECOIN_DIGEST": "externally-pinned-receipt",
        }
        original = base.copy()
        self.assertEqual(bench.phase_environment(base, distributed_wrapper=True), original)
        self.assertEqual(bench.phase_environment(base, recursive=True), original)
        outer = bench.phase_environment(base, recursive=True, distributed_wrapper=True)
        self.assertEqual(outer["MULTI_STARK_KZG_CUDA_RESIDENT_GIB"], "8")
        self.assertEqual(outer["MULTI_STARK_KZG_CUDA_DISTRIBUTED_QUOTIENT"], "1")
        self.assertEqual(outer["MULTI_STARK_KZG_CUDA_DISTRIBUTED_LOOKUP"], "1")
        self.assertEqual(outer["MULTI_STARK_KZG_SETUP"], "filecoin")
        self.assertEqual(outer["MULTI_STARK_KZG_FILECOIN_DIGEST"], original["MULTI_STARK_KZG_FILECOIN_DIGEST"])
        cpu = bench.phase_environment(base, cpu=True, recursive=True, distributed_wrapper=True)
        self.assertEqual(cpu, dict(original, MULTI_STARK_KZG_BACKEND="cpu"))
        self.assertEqual(base, original)
        self.assertEqual(bench.captured_environment(outer), outer)

    def test_scientific_notation_and_cumulative_coset_timers(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "profile.log"
            log.write_text(
                "\x1b[32mINFO\x1b[0m KZG quotient constraints evaluated rows=256 "
                "selector_seconds=7.377e-5 constraint_seconds=8.0621e-5\n"
                "INFO KZG quotient coset evaluated circuit=0 coset=0 lde_seconds=2 seconds=5\n"
                "INFO KZG quotient coset evaluated circuit=0 coset=1 lde_seconds=3 seconds=12\n"
            )
            result = bench.summarize_host_operations(log)
            constraints = result["KZG quotient constraints evaluated"]
            self.assertAlmostEqual(constraints["selector_seconds"], 0.00007377)
            self.assertAlmostEqual(constraints["constraint_seconds"], 0.000080621)
            self.assertEqual(result["KZG quotient coset evaluated"], {"calls": 2, "lde_seconds": 5})

    def test_old_transfer_logs_and_new_msm_compute_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "cuda.log"
            log.write_text(
                "KZG CUDA profile device=0 operation=msm-points elements=128 "
                "upload_ms=1 kernel_ms=0 download_ms=0 host_copy_ms=2 "
                "upload_bytes=12288 download_bytes=0 device_copy_bytes=0\n"
                "KZG CUDA profile device=1 operation=msm-compute elements=128 "
                "upload_ms=0 kernel_ms=3 download_ms=0 host_copy_ms=0 "
                "upload_bytes=0 download_bytes=0 device_copy_bytes=0 call_ms=5\n"
            )
            result = bench.summarize_cuda_events(log)
            qualified, parsed = bench.summarize_cuda_events(log, with_quality=True)
            self.assertEqual(result, qualified)
            self.assertTrue(parsed["totals_complete"])
            self.assertEqual(parsed["records_parsed"], 2)
            self.assertEqual(result["msm-points"]["upload_bytes"], 12288)
            self.assertNotIn("call_ms", result["msm-points"])
            self.assertEqual(result["msm-compute"]["kernel_ms"], 3)
            self.assertEqual(result["msm-compute"]["call_ms"], 5)

    def test_interleaved_cuda_record_is_skipped_without_partial_totals(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "cuda.log"
            clean = (
                "KZG CUDA profile device=0 operation=retain elements=128 "
                "upload_ms=1 kernel_ms=0 download_ms=0 host_copy_ms=2 "
                "upload_bytes=4096 download_bytes=0 device_copy_bytes=0 call_ms=3\n"
            )
            interleaved = clean.replace("host_copy_ms=2", "host_copy_ms=29.2026-10-09T22:01:17.918965Z")
            original = clean + interleaved + clean
            log.write_text(original)
            with self.assertRaisesRegex(ValueError, "line 2: invalid numeric fields: host_copy_ms"):
                bench.summarize_cuda_events(log)
            result, parsed = bench.summarize_cuda_events(log, with_quality=True)
            self.assertEqual(result["retain"]["calls"], 2)
            self.assertEqual(result["retain"]["upload_bytes"], 8192)
            self.assertEqual(result["retain"]["upload_ms"], 2)
            self.assertEqual(result["retain"]["host_copy_ms"], 4)
            self.assertFalse(parsed["totals_complete"])
            self.assertEqual(parsed["records_seen"], 3)
            self.assertEqual(parsed["records_parsed"], 2)
            self.assertEqual(parsed["records_skipped"], 1)
            self.assertEqual(parsed["skipped"], [{
                "line": 2, "reason": "invalid numeric fields: host_copy_ms",
                "text": interleaved.rstrip("\n"),
            }])
            self.assertEqual(log.read_text(), original)

    def test_baseline_rejects_changed_proof_packet_profile_and_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for stage in ("fri-to-kzg", "recursive"):
                target = root / stage / "kzg"
                target.mkdir(parents=True)
                for name in bench.PROOF_FILES:
                    (target / name).write_bytes(name.encode())
            report = {"status": "verified_regenerated_chain", "fri_input": {"outer-proof.bin": "abc"}}
            (root / "report.json").write_text(json.dumps(report))
            baseline = bench.load_baseline(root)
            target = root / "recursive/kzg"
            expected = baseline["proofs"]["recursive_kzg"]
            bench.require_hashes(bench.proof_hashes(target), expected, "recursive")
            for name in bench.PROOF_FILES:
                with self.subTest(name=name):
                    original = (target / name).read_bytes()
                    (target / name).write_bytes(b"x" * len(original))
                    with self.assertRaisesRegex(RuntimeError, name):
                        bench.require_hashes(bench.proof_hashes(target), expected, "recursive")
                    (target / name).write_bytes(original)
            with self.assertRaisesRegex(RuntimeError, "outer-proof.bin"):
                bench.require_hashes({"outer-proof.bin": "changed"}, baseline["fri_input"], "input")
            report["status"] = "failed"
            (root / "report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "verified"):
                bench.load_baseline(root)


if __name__ == "__main__":
    unittest.main()
