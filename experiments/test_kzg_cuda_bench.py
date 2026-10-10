import importlib.util
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location("bench", Path(__file__).with_name("kzg-cuda-bench.py"))
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def idle_evidence(devices=(0, 1, 2, 3)):
    memory = {
        "driver_free_bytes": 900, "total_bytes": 1000,
        "current_pool_reserved_bytes": 10, "current_pool_used_bytes": 10,
        "default_pool_reserved_bytes": 10, "default_pool_used_bytes": 10,
        "current_pool_is_default": True, "resident_coefficient_bytes": 0,
        "srs_point_bytes": 0, "msm_workspace_bytes": 0,
    }
    return {"cuda_enabled": True, "initialized": True, "quiesced": True, "devices": [
        {"device": device, "before": memory.copy(), "after": memory.copy(),
         "released_srs_point_bytes": 0, "released_msm_workspace_bytes": 0} for device in devices]}


def write_statement(path, second=1):
    path.write_bytes(b"".join(word.to_bytes(8, "little") for word in [1, 18, 0, second, *([0] * 16)]))


class BenchmarkEvidenceTests(unittest.TestCase):
    @staticmethod
    def capabilities(role):
        return {
            "format": "multi-stark-capabilities/v1", "binary": bench.CAPABILITY_BINARIES[role],
            "kzg": True, "kzg_cuda": True, "goldilocks_cuda": True, "parallel": True,
        }

    def accepted_report(self):
        receipt, ceremony = "3" * 64, "4" * 64
        first = {
            "setup": "filecoin", "development_srs": False, "setup_id": "1" * 64,
            "filecoin_manifest_digest": receipt, "ceremony_id": ceremony,
            "public_max_degree": (1 << 28) - 2, "trace_heights": [1 << 24, 2, 1 << 17],
            "proof_bytes": 49000, "packet_bytes": 49176,
            "native_verification_passed": True, "negative_tests_pass": True,
        }
        outer = dict(first, setup_id="2" * 64, trace_heights=[1 << 27, 1 << 17],
                     proof_bytes=1909, packet_bytes=2181, external_pairings_pass=True)
        phases = {"fri_compress": {"exit_code": 0, "backend": "cuda"}}
        for phase in ("fri_stage_and_prove", "recursive_stage_and_prove"):
            phases[phase] = {
                "exit_code": 0, "backend": "cuda",
                "cuda_profile_parse": {"records_skipped": 0, "devices": {
                    str(device): {"records": 2, "kernel_records": 1, "kernel_ms": 0.5}
                    for device in range(4)}},
            }
        for phase in ("fri_cpu_verify", "recursive_cpu_verify"):
            phases[phase] = {"exit_code": 0, "backend": "cpu",
                             "environment": {"MULTI_STARK_KZG_BACKEND": "cpu"}}
        return {
            "status": "verified_regenerated_chain", "setup": "filecoin", "fused_staging": True,
            "fri_compression_rerun": True, "fresh_output_directory": True,
            "timing_boundary": "aggregation root to verified final packet", "pipeline_wall_seconds": 299.9,
            "root_artifacts": {name: "5" * 64 for name in
                               ("root-vk.bin", "root-proof.bin", "root-claims.bin")},
            "capabilities": {role: self.capabilities(role) for role in bench.CAPABILITY_BINARIES},
            "kzg_devices": [0, 1, 2, 3], "phases": phases,
            "environment": {"MULTI_STARK_KZG_FILECOIN_DIGEST": receipt},
            "first_stage_verification": first, "recursive_verification": outer,
            "first_stage_cpu_verification": copy.deepcopy(first),
            "recursive_cpu_verification": copy.deepcopy(outer),
            "setup_bindings": {"fri_to_kzg": first["setup_id"], "recursive_kzg": outer["setup_id"]},
            "proofs": {stage: {
                "proof.compact.bin": {"sha256": "6" * 64, "bytes": summary["proof_bytes"]},
                "packet.bin": {"sha256": "7" * 64, "bytes": summary["packet_bytes"]},
                "profile-id.bin": {"sha256": "8" * 64, "bytes": 32},
            } for stage, summary in (("fri_to_kzg", first), ("recursive_kzg", outer))},
        }

    def test_actual_compiled_capabilities_reject_cpu_as_cuda(self):
        caps = self.capabilities("first_stage")
        caps["kzg_cuda"] = False
        with self.assertRaisesRegex(ValueError, "lacks compiled features: kzg_cuda"):
            bench.validate_capabilities(caps, "first_stage")
        with mock.patch.object(bench.subprocess, "run", return_value=mock.Mock(stdout=json.dumps(caps) + "\n")) as run:
            with self.assertRaisesRegex(ValueError, "kzg_cuda"):
                bench.read_capabilities(Path("/tmp/explicit-first-binary"), "first_stage",
                                        {"MULTI_STARK_KZG_BACKEND": "cuda"})
        self.assertEqual(run.call_args.args[0], ["/tmp/explicit-first-binary", "capabilities"])
        caps = self.capabilities("fri")
        caps["goldilocks_cuda"] = False
        with self.assertRaisesRegex(ValueError, "goldilocks_cuda"):
            bench.validate_capabilities(caps, "fri")
        for caps in ({}, dict(self.capabilities("wrapper"), kzg_cuda="true"), self.capabilities("first_stage")):
            with self.assertRaises(ValueError):
                bench.validate_capabilities(caps, "wrapper")
        caps = dict(self.capabilities("wrapper"), parallel=False)
        bench.validate_capabilities(caps, "wrapper")
        with self.assertRaisesRegex(ValueError, "parallel"):
            bench.validate_capabilities(caps, "wrapper", acceptance=True)

    def test_acceptance_requires_full_filecoin_boundary_and_explicit_four_devices(self):
        self.assertEqual(bench.parse_kzg_devices("3, 1,2,0"), [3, 1, 2, 0])
        for invalid in ("", "0,0,1,2", "-1,0,1,2", "0.0,1,2,3", "0,1,2,2147483648"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                bench.parse_kzg_devices(invalid)
        options = dict(acceptance=True, setup="filecoin", root_artifacts=Path("/root-input"), devices=[0, 1, 2, 3])
        bench.validate_acceptance_options(**options)
        for changed in ({"setup": "development"}, {"root_artifacts": None}, {"devices": None},
                        {"devices": [0]}, {"devices": [0, 0, 1, 2]}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                bench.validate_acceptance_options(**dict(options, **changed))
        bench.validate_acceptance_options(acceptance=False, setup="development", root_artifacts=None, devices=None)

    def test_goal_checks_actual_limits_separately_from_correctness(self):
        report = self.accepted_report()
        self.assertTrue(bench.evaluate_goal(report)["passed"])
        for mutation, expected in (
                (lambda r: r.update(pipeline_wall_seconds=300.0), "pipeline_under_300_seconds"),
                (lambda r: r.update(pipeline_wall_seconds=float("nan")), "pipeline_under_300_seconds"),
                (lambda r: r.update(fri_compression_rerun=False), "fresh_root_boundary"),
                (lambda r: r["recursive_verification"].update(trace_heights=[1 << 28, 1 << 17]), "single_wrapper_trace_cap"),
                (lambda r: r["recursive_verification"].update(trace_heights=[1 << 26, 1 << 26, 1 << 17]), "single_wrapper_trace_cap"),
                (lambda r: r["first_stage_verification"].update(trace_heights=[3]), "stage_one_trace_cap"),
                (lambda r: r["proofs"]["recursive_kzg"]["packet.bin"].update(bytes=3000), "final_packet_below_3000_bytes"),
                (lambda r: r["first_stage_verification"].update(development_srs=True), "both_stages_authenticated_filecoin"),
                (lambda r: r["recursive_cpu_verification"].update(filecoin_manifest_digest="a" * 64), "both_stages_authenticated_filecoin"),
                (lambda r: r["recursive_verification"].update(ceremony_id="b" * 64), "both_stages_authenticated_filecoin"),
                (lambda r: r["recursive_cpu_verification"].update(external_pairings_pass=False), "native_verification_and_negative_checks"),
                (lambda r: r["phases"]["fri_cpu_verify"].update(backend="cuda"), "independent_cpu_verification"),
                (lambda r: r["capabilities"]["wrapper"].update(kzg_cuda=False), "compiled_cuda_capabilities"),
                (lambda r: r["setup_bindings"].update(fri_to_kzg="b" * 64), "fri_to_kzg_artifacts_and_profile")):
            with self.subTest(expected=expected):
                changed = copy.deepcopy(report)
                mutation(changed)
                goal = bench.evaluate_goal(changed)
                self.assertFalse(goal["passed"])
                self.assertIn(expected, goal["failures"])
                self.assertEqual(changed["status"], "verified_regenerated_chain")
        self.assertFalse(bench.evaluate_goal({})["passed"])

    def test_unfused_acceptance_still_requires_both_fresh_witness_stages(self):
        report = self.accepted_report()
        report["fused_staging"] = False
        for prefix in ("fri", "recursive"):
            report["phases"][f"{prefix}_prove"] = report["phases"].pop(f"{prefix}_stage_and_prove")
            report["phases"][f"{prefix}_stage"] = {"exit_code": 0, "backend": "cuda"}
        self.assertTrue(bench.evaluate_goal(report)["passed"])
        report["phases"].pop("fri_stage")
        self.assertIn("fresh_kzg_witness_stages", bench.evaluate_goal(report)["failures"])

    def test_goal_requires_per_device_kernels_not_flags_or_utilization(self):
        report = self.accepted_report()
        for phase, gate in (("fri_stage_and_prove", "fri_to_kzg_four_device_kernels"),
                            ("recursive_stage_and_prove", "recursive_kzg_four_device_kernels")):
            changed = copy.deepcopy(report)
            changed["phases"][phase]["cuda_profile_parse"]["devices"].pop("3")
            changed["gpu_samples_every_2_seconds"] = {"3": {"peak_utilization_percent": 100}}
            changed["environment"]["MULTI_STARK_KZG_CUDA_DEVICES"] = "0,1,2,3"
            self.assertIn(gate, bench.evaluate_goal(changed)["failures"])
            changed = copy.deepcopy(report)
            changed["phases"][phase]["cuda_profile_parse"]["devices"]["3"] = {
                "records": 12, "kernel_records": 0, "kernel_ms": 0,
            }
            self.assertIn(gate, bench.evaluate_goal(changed)["failures"])

    def test_artifacts_are_recorded_without_a_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name in bench.PROOF_FILES:
                (path / name).write_bytes(name.encode())
            report = {"parity": {}}
            for stage in ("fri_to_kzg", "recursive_kzg"):
                bench.record_proof_evidence(report, stage, path)
                self.assertEqual(report["proofs"][stage], bench.proof_hashes(path))
            self.assertEqual(report["parity"], {})

    def retained_report(self):
        report = self.accepted_report()
        report.update(retained_workers=True, latency_boundary="retained_worker_fresh_request")
        report["fri_input"] = {name: "b" * 64 for name in bench.FRI_FILES}
        startup = {"completed": True, "wall_seconds": 101, "workers": {}, "phases": {},
                   "seed": {"expected_claims_sha256": "c" * 64,
                            "input_hashes": {name: "d" * 64 for name in bench.FRI_FILES}},
                   "proofs": {"fri_to_kzg": {"proof.compact.bin": {"sha256": "e" * 64}}}}
        report["worker_startup"] = startup
        for role, stage, phase in (("first_stage", "fri_to_kzg", "fri_stage_and_prove"),
                                   ("wrapper", "recursive_kzg", "recursive_stage_and_prove")):
            identity = {"frontend": "f" * 64, "setup": report["setup_bindings"][stage], "profile": "a" * 64}
            startup["workers"][role] = {
                "pid": 42,
                "listening": {"format": bench.WORKER_FORMAT, "binary": bench.CAPABILITY_BINARIES[role],
                              "id": None, "status": "listening", "prepared": False},
                "shutdown": {"exit_code": 0},
            }
            for reused in (False, True):
                scope = report if reused else startup
                name = phase if reused else f"startup_{phase}"
                request_id = f"{'fresh' if reused else 'warmup'}-{role}"
                destination = f"/output/{request_id}"
                scope["phases"].setdefault(name, {}).update({
                    "exit_code": 0, "backend": "cuda", "worker_pid": 42,
                    "worker_log_start_byte": 200 if reused else 0,
                    "worker_log_end_byte": 400 if reused else 200,
                    "output_identity": dict(identity, profile="b" * 64) if reused and role == "first_stage"
                                       else identity.copy(),
                    "worker_request": {"id": request_id, "output": destination},
                    "worker_response": {
                        "format": bench.WORKER_FORMAT, "binary": bench.CAPABILITY_BINARIES[role],
                        "status": "proved", "prepared": True, "id": request_id, "output": destination,
                        "frontend_reused": reused, "loaded_key_reused": reused, "idle": idle_evidence(),
                    },
                    "expected_claims_sha256": "5" * 64 if reused else "c" * 64,
                    "input_hashes": (report["fri_input"] if reused else startup["seed"]["input_hashes"])
                    if role == "first_stage" else {"proof.compact.bin": "6" * 64 if reused else "e" * 64},
                })
        return report

    def test_retained_acceptance_requires_actual_preparation_reuse_and_fresh_evidence(self):
        report = self.retained_report()
        self.assertTrue(bench.evaluate_goal(report)["passed"])
        phase = "fri_stage_and_prove"
        for mutation in (
                lambda r: r.update(latency_boundary="cold_process_pipeline"),
                lambda r: r["worker_startup"].update(completed=False),
                lambda r: r["worker_startup"]["workers"]["first_stage"]["listening"].update(prepared=True),
                lambda r: r["worker_startup"]["workers"]["wrapper"]["shutdown"].update(exit_code=-9),
                lambda r: r["phases"][phase]["worker_response"].update(loaded_key_reused=False),
                lambda r: r["phases"][phase]["worker_response"]["idle"].update(quiesced=False),
                lambda r: r["phases"][phase]["worker_response"]["idle"]["devices"].pop(),
                lambda r: r["phases"][phase]["worker_response"]["idle"]["devices"][0]["after"].update(srs_point_bytes=1),
                lambda r: r["phases"][phase]["output_identity"].update(frontend="9" * 64),
                lambda r: r["phases"][phase].update(worker_log_start_byte=0),
                lambda r: r["phases"][phase].update(expected_claims_sha256="c" * 64),
                lambda r: r["phases"][phase].update(input_hashes=r["worker_startup"]["seed"]["input_hashes"]),
                lambda r: r["phases"]["recursive_stage_and_prove"]["input_hashes"].update(**{"proof.compact.bin": "e" * 64})):
            changed = copy.deepcopy(report)
            mutation(changed)
            self.assertIn("retained_worker_preparation_and_reuse", bench.evaluate_goal(changed)["failures"])
        changed = copy.deepcopy(report)
        changed["phases"][phase]["cuda_profile_parse"]["devices"].pop("3")
        changed["worker_startup"]["phases"][f"startup_{phase}"]["cuda_profile_parse"] = (
            report["phases"][phase]["cuda_profile_parse"])
        self.assertIn("fri_to_kzg_four_device_kernels", bench.evaluate_goal(changed)["failures"])

    def test_retained_profiles_follow_statement_and_stage_semantics(self):
        report = self.retained_report()
        self.assertTrue(bench.evaluate_goal(report)["passed"])
        startup = report["worker_startup"]
        startup["seed"]["expected_claims_sha256"] = "5" * 64
        for phase in startup["phases"].values():
            phase["expected_claims_sha256"] = "5" * 64
        self.assertIn("retained_worker_preparation_and_reuse", bench.evaluate_goal(report)["failures"])
        report["phases"]["fri_stage_and_prove"]["output_identity"]["profile"] = "a" * 64
        self.assertTrue(bench.evaluate_goal(report)["passed"])
        report = self.retained_report()
        report["phases"]["recursive_stage_and_prove"]["output_identity"]["profile"] = "b" * 64
        self.assertIn("retained_worker_preparation_and_reuse", bench.evaluate_goal(report)["failures"])

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
            self.assertEqual(parsed["devices"], {
                "0": {"records": 1, "kernel_records": 0, "kernel_ms": 0},
                "1": {"records": 1, "kernel_records": 1, "kernel_ms": 3},
            })

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

    def test_malformed_device_records_cannot_establish_kernel_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "cuda.log"
            log.write_text(
                "KZG CUDA profile device=3.5 operation=msm-compute elements=128 "
                "upload_ms=0 kernel_ms=10 download_ms=0 host_copy_ms=0 "
                "upload_bytes=0 download_bytes=0 device_copy_bytes=0\n"
            )
            operations, quality = bench.summarize_cuda_events(log, with_quality=True)
            self.assertEqual(operations, {})
            self.assertEqual(quality["devices"], {})
            self.assertEqual(quality["records_skipped"], 1)
            self.assertFalse(quality["totals_complete"])

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


class RetainedWorkerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "seed"
        self.source.mkdir()
        for name in bench.FRI_FILES:
            (self.source / name).write_bytes(name.encode())
        self.claims = self.root / "expected.bin"
        write_statement(self.claims)
        self.binary = self.root / "fake-worker"
        self.binary.write_text(f"#!{sys.executable}\n" + r'''
import hashlib
import json
import os
from pathlib import Path
import sys

responses = Path(sys.argv[2]).open("x")
mode = os.environ.get("WORKER_TEST_MODE", "normal")
idle = json.loads(os.environ["WORKER_TEST_IDLE"])
envelope = {"format": "multi-stark-worker/v1", "binary": "init_fri_kzg_prove"}
def emit(value):
    responses.write(json.dumps(dict(envelope, **value)) + "\n")
    responses.flush()
emit({"status": "listening", "id": None, "prepared": mode == "bad-ready"})
for index, line in enumerate(sys.stdin):
    request = json.loads(line)
    identity = request["id"] if mode != "wrong-id" else "other-id"
    if mode == "exit":
        sys.exit(3)
    if mode == "hang":
        continue
    if mode == "malformed":
        responses.write('{"format":"multi-stark-worker/v1","format":"duplicate"}\n')
        responses.flush()
        continue
    if mode == "failed":
        emit({"status": "failed", "id": identity, "error": "fixture rejection"})
        sys.exit(1)
    emit({"status": "request_started", "id": identity, "output": request["output"],
          "prepared": bool(index) if mode != "bad-start" else not bool(index)})
    print("KZG CUDA profile device=0 operation=" + ("fresh" if index else "warmup") +
          " elements=1 upload_ms=0 kernel_ms=1 download_ms=0 host_copy_ms=0 "
          "upload_bytes=0 download_bytes=0 device_copy_bytes=0", flush=True)
    output = Path(request["output"])
    (output / "kzg").mkdir(parents=True)
    (output / "plan-id.bin").write_bytes((b"x" if mode == "changed-frontend" and index else b"p") * 32)
    (output / "kzg-setup-id.bin").write_bytes((b"x" if mode == "changed-setup" and index else b"s") * 32)
    (output / "manifest.bin").write_bytes(b"fresh manifest")
    (output / "kzg/setup-0.bin").write_bytes(b"fresh verifier key")
    profile = hashlib.sha256(Path(request["expected_claims"]).read_bytes()).digest()
    if mode == "profile-drift" and index:
        profile = b"x" * 32
    (output / "kzg/profile-id.bin").write_bytes(profile)
    (output / "kzg/proof.compact.bin").write_bytes(b"fresh proof")
    (output / "kzg/packet.bin").write_bytes(b"fresh packet")
    emit({"status": "proved", "id": identity, "output": request["output"], "prepared": True,
          "frontend_reused": bool(index), "loaded_key_reused": bool(index),
          "preparation_seconds": 0 if index else 0.001, "request_seconds": 0.002, "idle": idle})
''')
        self.binary.chmod(0o755)

    def client(self, mode="normal", *, timeout=5):
        env = dict(os.environ, WORKER_TEST_MODE=mode, WORKER_TEST_IDLE=json.dumps(idle_evidence((0,))))
        worker = bench.WorkerClient(self.binary, "first_stage", self.root / f"worker-{mode}", env, [0], timeout=timeout)
        self.addCleanup(worker.close, graceful=False)
        return worker

    def request(self, worker, request_id="warmup", *, reused=False):
        return worker.request(request_id, self.source, self.root / request_id, self.claims,
                              self.root / f"{request_id}.log", reused=reused)

    def test_listening_is_not_preparation_and_request_logs_exclude_warmup(self):
        worker = self.client()
        self.assertFalse(worker.startup["listening"]["prepared"])
        first = self.request(worker)
        second_claims = self.root / "second-expected.bin"
        write_statement(second_claims, second=2)
        second = worker.request("fresh", self.source, self.root / "fresh", second_claims,
                                self.root / "fresh.log", reused=True)
        self.assertEqual(first["worker_pid"], second["worker_pid"])
        for key in ("frontend", "setup"):
            self.assertEqual(first["output_identity"][key], second["output_identity"][key])
        self.assertNotEqual(first["output_identity"]["profile"], second["output_identity"]["profile"])
        self.assertNotEqual(first["expected_claims_sha256"], second["expected_claims_sha256"])
        self.assertEqual(set(first["cuda_operation_totals_overlap"]), {"warmup"})
        self.assertEqual(set(second["cuda_operation_totals_overlap"]), {"fresh"})
        self.assertEqual(first["worker_log_end_byte"], second["worker_log_start_byte"])
        self.assertEqual(second["worker_response"]["preparation_seconds"], 0)
        self.assertGreater(second["wall_seconds"], 0)
        self.assertNotIn("peak_rss_bytes", second)
        third = self.request(worker, "fresh-a-again", reused=True)
        self.assertEqual(first["output_identity"], third["output_identity"])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.request(worker, "fresh", reused=True)
        worker.close()
        self.assertEqual(worker.process.poll(), 0)
        self.assertEqual(worker.shutdown["exit_code"], 0)

    def test_worker_seed_snapshot_is_independent_and_export_uses_generated_inner_proof(self):
        fri, claims, evidence = bench.snapshot_worker_seed(self.source, self.claims, self.root / "snapshot")
        self.assertEqual(evidence["input_hashes"], bench.worker_input_hashes("first_stage", fri))
        original_claims = claims.read_bytes()
        write_statement(self.claims, second=2)
        (self.source / "outer-proof.bin").write_bytes(b"changed")
        self.assertEqual(claims.read_bytes(), original_claims)
        self.assertEqual((fri / "outer-proof.bin").read_bytes(), b"outer-proof.bin")
        worker = self.client()
        self.request(worker)
        (self.root / "warmup/setup-spoof.bin").write_bytes(b"not a generated key")
        inner = self.root / "intermediate"
        bench.export_inner_proof(self.root / "warmup", inner)
        self.assertEqual((inner / "proof.compact.bin").read_bytes(), b"fresh proof")
        self.assertEqual((inner / "setup-0.bin").read_bytes(), b"fresh verifier key")
        self.assertFalse((inner / "setup-spoof.bin").exists())
        self.assertEqual(set(bench.worker_input_hashes("wrapper", inner)),
                         {"manifest.bin", "kzg-setup-id.bin", "setup-0.bin", "proof.compact.bin"})

    def test_worker_rejects_prepared_listening_and_bad_request_channels(self):
        with self.assertRaisesRegex(ValueError, "unprepared"):
            self.client("bad-ready")
        for mode, pattern in (("wrong-id", "expected request id"), ("bad-start", "expected request id"),
                              ("malformed", "duplicate"), ("failed", "fixture rejection"),
                              ("exit", "exited 3")):
            with self.subTest(mode=mode):
                worker = self.client(mode)
                with self.assertRaisesRegex((ValueError, RuntimeError), pattern):
                    self.request(worker, mode)
                self.assertIsNone(worker.last_request["exit_code"])
                worker.close(graceful=False)
                self.assertIsNotNone(worker.process.poll())

    def test_profile_replacement_timeout_and_existing_output_fail_closed(self):
        for mode in ("changed-frontend", "changed-setup", "profile-drift"):
            with self.subTest(mode=mode):
                worker = self.client(mode)
                self.request(worker, f"{mode}-warmup")
                with self.assertRaisesRegex(ValueError, "profile changed"):
                    self.request(worker, f"{mode}-fresh", reused=True)
                worker.close(graceful=False)
        worker = self.client("hang", timeout=0.2)
        with self.assertRaisesRegex(TimeoutError, "timed out"):
            self.request(worker, "timeout")
        worker.close(graceful=False)
        self.assertIsNotNone(worker.process.poll())
        worker = self.client()
        with self.assertRaisesRegex(ValueError, "must be new"):
            worker.request("exists", self.source, self.source, self.claims,
                           self.root / "exists.log", reused=False)

    def test_interrupted_graceful_shutdown_reaps_the_worker(self):
        worker = self.client()
        original_wait = worker.process.wait
        calls = 0

        def interrupted_wait(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise KeyboardInterrupt()
            return original_wait(*args, **kwargs)

        with mock.patch.object(worker.process, "wait", side_effect=interrupted_wait):
            with self.assertRaises(KeyboardInterrupt):
                worker.close()
        self.assertIsNotNone(worker.process.poll())
        self.assertTrue(worker.log.closed)

    def test_retained_options_and_canonical_statement_admission(self):
        args = dict(retained=True, root_artifacts=self.root, seed_fri=self.source, seed_claims=self.claims)
        bench.validate_worker_options(**args)
        for mutation in ({"root_artifacts": None}, {"seed_fri": None}, {"seed_claims": None}, {"retained": False}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bench.validate_worker_options(**dict(args, **mutation))
        original = bench.expected_statement_bytes(self.claims)
        for malformed in (original[:-1], original + b"x", b"x" * 100000,
                          (2).to_bytes(8, "little") + original[8:],
                          original[:16] + (1).to_bytes(8, "little") + original[24:],
                          original[:24] + (0xffffffff00000001).to_bytes(8, "little") + original[32:]):
            self.claims.write_bytes(malformed)
            with self.assertRaises(ValueError):
                bench.expected_statement_bytes(self.claims)

    def test_development_public_degree_requires_explicit_diagnostic_selection(self):
        options = dict(enabled=True, setup="development", retained=True, acceptance=False)
        bench.validate_development_public_degree_options(**options)
        for mutation in ({"setup": "filecoin"}, {"retained": False}, {"acceptance": True}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bench.validate_development_public_degree_options(**dict(options, **mutation))
        report = dict(setup="development", development_srs=True, known_trapdoor=True,
                      filecoin_acceptance=False, ceremony_id=None, filecoin_manifest_digest=None,
                      public_max_degree=(1 << 28) - 2, public_setup_id="a" * 64)
        bench.validate_development_public_degree_reports([report, dict(report)])
        for mutation in ({"setup": "filecoin"}, {"development_srs": False}, {"known_trapdoor": False},
                         {"filecoin_acceptance": True}, {"ceremony_id": "a" * 64},
                         {"filecoin_manifest_digest": "a" * 64}, {"public_max_degree": 2**24 - 1},
                         {"public_setup_id": "b" * 64}, {"public_setup_id": None}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                bench.validate_development_public_degree_reports([report, dict(report, **mutation)])
        with self.assertRaises(ValueError):
            bench.validate_development_public_degree_reports([])

    def test_idle_release_retains_only_explicitly_reported_noncache_allocations(self):
        idle = idle_evidence((0,))
        bench.validate_worker_idle(idle, [0])
        for mutation in (lambda i: i.update(initialized=False), lambda i: i.update(devices=[]),
                         lambda i: i["devices"].append(copy.deepcopy(i["devices"][0])),
                         lambda i: i["devices"][0]["after"].update(resident_coefficient_bytes=1),
                         lambda i: i["devices"][0]["after"].update(msm_workspace_bytes=1),
                         lambda i: i["devices"][0]["after"].update(current_pool_used_bytes=11),
                         lambda i: i["devices"][0]["after"].update(driver_free_bytes=-1)):
            changed = copy.deepcopy(idle)
            mutation(changed)
            with self.assertRaises(ValueError):
                bench.validate_worker_idle(changed, [0])
        with self.assertRaisesRegex(ValueError, "selected KZG devices"):
            bench.validate_worker_idle(idle, [0, 1])

    def test_fake_end_to_end_retained_driver_keeps_startup_outside_fresh_boundary(self):
        self.exercise_fake_pipeline()

    def test_fake_development_v4_pipeline_uses_diagnostic_workers_and_cpu_verifiers(self):
        self.exercise_fake_pipeline(diagnostic=True)

    def exercise_fake_pipeline(self, diagnostic=False):
        binary = self.root / "fake-pipeline"
        events = self.root / "events.jsonl"
        binary.write_text(f"#!{sys.executable}\n" + r'''
import hashlib
import json
import os
from pathlib import Path
import sys

role = Path(sys.argv[0]).name
def event(kind, **values):
    with Path(os.environ["WORKER_TEST_EVENTS"]).open("a") as events:
        events.write(json.dumps(dict(binary=role, kind=kind, **values)) + "\n")
if sys.argv[1] == "capabilities":
    print(json.dumps(dict(format="multi-stark-capabilities/v1", binary=role,
                         kzg=True, kzg_cuda=True, goldilocks_cuda=True, parallel=True)))
elif sys.argv[1] in ("verify", "dev-v4-verify"):
    event("verify", command=sys.argv[1])
    target = Path(sys.argv[2]) / "kzg"
    (target / "verify-report.json").write_bytes((target / "prove-report.json").read_bytes())
elif role == "ix_root":
    event("fresh-fri")
    target = Path(sys.argv[2])
    target.mkdir()
    for name in ("outer-vk.bin", "outer-proof.bin", "outer-claims.bin"):
        (target / name).write_bytes(("fresh " + name).encode())
else:
    assert sys.argv[1] in ("serve", "dev-v4-serve")
    event("serve", command=sys.argv[1])
    responses = Path(sys.argv[2]).open("x")
    def emit(**values):
        responses.write(json.dumps(dict(format="multi-stark-worker/v1", binary=role, **values)) + "\n")
        responses.flush()
    emit(status="listening", id=None, prepared=False)
    for index, line in enumerate(sys.stdin):
        request = json.loads(line)
        event("request", id=request["id"], input=request["input"], output=request["output"])
        emit(status="request_started", id=request["id"], prepared=bool(index), output=request["output"])
        for device in range(4):
            print(f"KZG CUDA profile device={device} operation=msm-compute elements=1 "
                  "upload_ms=0 kernel_ms=1 download_ms=0 host_copy_ms=0 "
                  "upload_bytes=0 download_bytes=0 device_copy_bytes=0", flush=True)
        target = Path(request["output"])
        (target / "kzg").mkdir(parents=True)
        first = role == "init_fri_kzg_prove"
        setup = (b"1" if first else b"2") * 32
        (target / ("plan-id.bin" if first else "frontend-id.bin")).write_bytes(b"p" * 32)
        (target / "kzg-setup-id.bin").write_bytes(setup)
        (target / "manifest.bin").write_bytes(b"manifest")
        (target / "kzg/setup-0.bin").write_bytes(b"fixed key")
        profile = hashlib.sha256(Path(request["expected_claims"]).read_bytes()).digest() if first else b"v" * 32
        (target / "kzg/profile-id.bin").write_bytes(profile)
        proof = request["id"].encode()
        (target / "kzg/proof.compact.bin").write_bytes(proof)
        (target / "kzg/packet.bin").write_bytes(b"packet:" + proof)
        native = dict(setup="filecoin", development_srs=False, setup_id=setup.hex(),
                      filecoin_manifest_digest="3" * 64, ceremony_id="4" * 64,
                      public_max_degree=(1 << 28) - 2, trace_heights=[2, 2],
                      proof_bytes=len(proof), packet_bytes=len(proof) + 7,
                      native_verification_passed=True, negative_tests_pass=True, external_pairings_pass=True)
        if sys.argv[1] == "dev-v4-serve":
            native.update(setup="development", development_srs=True, known_trapdoor=True,
                          filecoin_acceptance=False, ceremony_id=None, filecoin_manifest_digest=None,
                          public_setup_id="5" * 64)
        (target / "kzg/prove-report.json").write_text(json.dumps(native))
        emit(status="proved", id=request["id"], output=request["output"], prepared=True,
             frontend_reused=bool(index), loaded_key_reused=bool(index), preparation_seconds=0,
             request_seconds=0.001, idle=json.loads(os.environ["WORKER_TEST_IDLE"]))
''')
        binary.chmod(0o755)
        root_input = self.root / "root-input"
        root_input.mkdir()
        for name in ("root-vk.bin", "root-proof.bin"):
            (root_input / name).write_bytes(name.encode())
        write_statement(root_input / "root-claims.bin", second=2)
        output = self.root / "result"
        selection = (["--development-public-degree", "--setup", "development"] if diagnostic else
                     ["--acceptance", "--setup", "filecoin", "--filecoin-cache", str(self.root / "cache"),
                      "--filecoin-digest", "3" * 64])
        arguments = [str(bench.__file__), *selection, "--kzg-devices", "0,1,2,3",
                     "--retained-workers", "--worker-seed-fri", str(self.source),
                     "--worker-seed-claims", str(self.claims), "--root-artifacts", str(root_input),
                     "--first-stage-binary", str(binary), "--wrapper-binary", str(binary),
                     "--fri-binary", str(binary), "--output", str(output)]
        original_popen = bench.subprocess.Popen
        monitor = mock.Mock()

        def spawn(command, *args, **kwargs):
            return monitor if command[0] == "nvidia-smi" else original_popen(command, *args, **kwargs)

        def hardware(command, **kwargs):
            return "{}" if command[0] == "lscpu" else "fixture\n"

        old_signal = bench.signal.getsignal(bench.signal.SIGTERM)
        try:
            with mock.patch.object(sys, "argv", arguments), \
                    mock.patch.dict(os.environ, {"WORKER_TEST_IDLE": json.dumps(idle_evidence()),
                                                 "WORKER_TEST_EVENTS": str(events)}, clear=True), \
                    mock.patch.object(bench.subprocess, "Popen", side_effect=spawn), \
                    mock.patch.object(bench.subprocess, "check_output", side_effect=hardware), \
                    mock.patch.object(bench, "benchmark_sources", return_value=[Path(bench.__file__)]), \
                    mock.patch("builtins.print"):
                bench.main()
        finally:
            bench.signal.signal(bench.signal.SIGTERM, old_signal)
        report = json.loads((output / "report.json").read_text())
        self.assertEqual(report["goal_acceptance"]["passed"], not diagnostic)
        self.assertEqual(report["development_public_degree"], diagnostic)
        self.assertEqual(report["latency_boundary"], "retained_worker_fresh_request")
        self.assertTrue(report["worker_startup"]["completed"])
        self.assertGreater(report["worker_startup_plus_pipeline_wall_seconds"], report["pipeline_wall_seconds"])
        self.assertEqual(len(report["worker_startup"]["phases"]), 2)
        events = [json.loads(line) for line in events.read_text().splitlines()]
        order = [entry["id"] if entry["kind"] == "request" else entry["kind"] for entry in events]
        self.assertEqual(order, ["serve", "warmup-first_stage", "serve", "warmup-wrapper", "fresh-fri",
                                 "fresh-first_stage", "fresh-wrapper", "verify", "verify"])
        for event in events:
            if event["kind"] in ("serve", "verify"):
                self.assertEqual(event["command"], ("dev-v4-" if diagnostic else "") + event["kind"])
        if diagnostic:
            self.assertEqual(bench.evaluate_goal(report)["failures"], ["both_stages_authenticated_filecoin"])
        self.assertEqual(report["phases"]["recursive_stage_and_prove"]["input_hashes"]["proof.compact.bin"],
                         report["proofs"]["fri_to_kzg"]["proof.compact.bin"]["sha256"])
        self.assertNotEqual(report["worker_startup"]["proofs"]["fri_to_kzg"], report["proofs"]["fri_to_kzg"])
        self.assertTrue(all(worker["shutdown"]["exit_code"] == 0
                            for worker in report["worker_startup"]["workers"].values()))


if __name__ == "__main__":
    unittest.main()
