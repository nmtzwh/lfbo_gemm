import os
import json
import subprocess
import tempfile
import threading
import unittest
from pathlib import Path

from matopt.protocol import MeasurementProfile, Workload, canonical_json, request
from matopt.runner import MatOptRunner
from matopt.search.lfbo import LFBOConfig
from matopt.session import TuningSession
from matopt.space import PlanSpace
from matopt.space_config import SpaceConfig


@unittest.skipUnless(os.environ.get("MATOPT_RUNNER"), "MATOPT_RUNNER is not set")
class NativeAVX2Tests(unittest.TestCase):
    def setUp(self):
        self.runner = MatOptRunner(
            os.environ["MATOPT_RUNNER"], env={"ONEDNN_MAX_CPU_ISA": "AVX2"}
        )
        self.workload = Workload(64, 64, 64, 1, os.environ.get("MATOPT_CPU", "0"))
        self.profile = MeasurementProfile(warmups=1, samples=1, minimum_sample_ms=10)

    def test_capabilities_baseline_and_evaluate(self):
        caps = self.runner.capabilities(self.workload)
        self.assertEqual(caps["status"], "capabilities")
        self.assertIn("avx2", caps["effective_isa"].lower())
        baseline = self.runner.baseline(
            self.workload, self.profile, caps["fingerprint"]
        )
        self.assertEqual(baseline["status"], "benchmarked")
        result = self.runner.evaluate(
            self.workload,
            baseline["effective_plan"],
            self.profile,
            caps["fingerprint"],
        )
        self.assertEqual(result["status"], "benchmarked")
        self.assertTrue(result["measurement"]["correct"])
        inspected = self.runner.inspect(
            self.workload, baseline["effective_plan"], caps["fingerprint"]
        )
        self.assertEqual(inspected["status"], "accepted")
        invalid = dict(baseline["effective_plan"], nthr_k=2)
        rejected = self.runner.evaluate(
            self.workload, invalid, self.profile, caps["fingerprint"]
        )
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(rejected["reason_code"], "invalid_thread_partition")

    def test_lfbo_session(self):
        with tempfile.TemporaryDirectory() as directory:
            result = TuningSession(
                self.workload,
                self.runner,
                history=Path(directory) / "history.jsonl",
            ).tune(
                budget=4,
                algorithm="lfbo",
                lfbo_config=LFBOConfig(
                    initial_population=2,
                    copies=1,
                    generations=2,
                    neighbors=8,
                    fraction_selected=0.5,
                    patience=2,
                    trees=8,
                ),
                measurement=self.profile,
            )
            self.assertEqual(result["search"]["algorithm"], "lfbo_pattern_search")
            self.assertIn("one_shot", result["selected"])

    def test_custom_m_chunk_size_is_realized(self):
        caps = self.runner.capabilities(self.workload)
        baseline = self.runner.baseline(
            self.workload, self.profile, caps["fingerprint"]
        )
        space = PlanSpace(
            self.workload,
            baseline["effective_plan"],
            caps,
            SpaceConfig.from_dict(
                {
                    "inherit_baseline": False,
                    "domains": {"M_chunk_size": [2]},
                }
            ),
        )
        plan = space.canonicalize({"M_chunk_size": 2})
        result = self.runner.evaluate(
            self.workload, plan, self.profile, caps["fingerprint"]
        )
        self.assertEqual(result["status"], "benchmarked")
        self.assertEqual(result["finalized"]["plan"]["M_chunk_size"], 2)

    def test_aot_capture_replays_copied_images(self):
        caps = self.runner.capabilities(self.workload)
        baseline = self.runner.baseline(
            self.workload, self.profile, caps["fingerprint"]
        )
        with tempfile.TemporaryDirectory() as directory:
            captured = self.runner.capture_aot(
                self.workload,
                baseline["effective_plan"],
                caps["fingerprint"],
                directory,
            )
            self.assertEqual(captured["status"], "captured")
            self.assertTrue(captured["aot_bundle"]["copied_address_replay"])
            self.assertTrue(captured["aot_bundle"]["images"])
            for image in captured["aot_bundle"]["images"]:
                self.assertEqual(len(image["sha256"]), 64)
                self.assertTrue((Path(directory) / image["file"]).is_file())
        persistent = dict(
            baseline["effective_plan"], N_blk=32, pack_b="persistent_n32"
        )
        with tempfile.TemporaryDirectory() as directory:
            captured = self.runner.capture_aot(
                self.workload,
                persistent,
                caps["fingerprint"],
                directory,
            )
            self.assertEqual(captured["status"], "captured")
            self.assertIn(
                "reorder",
                {image["group"] for image in captured["aot_bundle"]["images"]},
            )

    def test_perf_diag_trace_and_control_window(self):
        caps = self.runner.capabilities(self.workload)
        baseline = self.runner.baseline(
            self.workload, self.profile, caps["fingerprint"]
        )
        payload = request(
            "native-perf-diag", self.workload,
            plan=baseline["effective_plan"],
            expected_fingerprint=caps["fingerprint"],
        )
        with tempfile.TemporaryDirectory() as directory:
            request_path = Path(directory) / "request.json"
            request_path.write_text(canonical_json(payload), encoding="utf-8")
            control_read, control_write = os.pipe()
            ack_read, ack_write = os.pipe()
            commands = []

            def acknowledge():
                with os.fdopen(control_read, "r", encoding="ascii") as control:
                    for line in control:
                        commands.append(line.strip())
                        os.write(ack_write, b"ack\n")
                        if line.strip() == "disable":
                            break

            thread = threading.Thread(target=acknowledge)
            thread.start()
            completed = subprocess.run(
                [
                    self.runner.executable, "perf-diag",
                    "--request", str(request_path), "--stage", "full_driver",
                    "--perf-control-fd", str(control_write),
                    "--perf-ack-fd", str(ack_read),
                ],
                text=True, capture_output=True, check=False,
                pass_fds=(control_write, ack_read),
                env={**os.environ, "ONEDNN_MAX_CPU_ISA": "AVX2"},
            )
            os.close(control_write)
            os.close(ack_read)
            thread.join()
            os.close(ack_write)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        response = json.loads(completed.stdout)
        self.assertEqual(commands, ["enable", "disable"])
        self.assertEqual(response["status"], "diagnosed")
        self.assertEqual(response["trace"]["fidelity"], "exact_worker_trace")
        self.assertEqual(
            response["trace"]["flops"],
            2 * self.workload.m * self.workload.n * self.workload.k,
        )
        self.assertTrue(response["trace"]["entries"])


if __name__ == "__main__":
    unittest.main()
