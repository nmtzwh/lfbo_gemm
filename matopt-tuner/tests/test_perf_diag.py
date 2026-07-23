import json
import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest import mock

from matopt.perf_diag import (
    PMUProfile,
    PerfDiagError,
    _atomic_publish,
    attribute,
    confidence,
    parse_perf_csv,
)
from matopt.protocol import Workload
from matopt.runner import MatOptRunner


class PerfDiagTests(unittest.TestCase):
    def test_profile_requires_all_semantic_roles(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.yaml"
            path.write_text(
                "schema: perf_diag_profile_v1\n"
                "architecture: x86_64\ncpu_model_regex: '.*'\n"
                "isa: AVX2\nvector_bits: 256\nhomogeneous_cores: true\n"
                "fp32_flops_per_cycle_per_core: 16\nevents: {cycles: cycles}\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PerfDiagError, "missing roles"):
                PMUProfile.load(path)

    def test_perf_csv_preserves_availability_and_ratio(self):
        text = "1000;;cycles;100;98\n<not supported>;;bad;0;0\n"
        got = parse_perf_csv(text, {"cycles": "cycles", "bad": "l1_refills"})
        self.assertEqual(got["cycles"]["value"], 1000)
        self.assertAlmostEqual(got["cycles"]["running_ratio"], 0.98)
        self.assertFalse(got["l1_refills"]["available"])

    def test_confidence(self):
        got = confidence([1.0, 1.0, 1.0])
        self.assertEqual(got["median"], 1.0)
        self.assertEqual(got["cv"], 0.0)

    def test_waterfall_closes_and_keeps_negative_components(self):
        got = attribute(
            workload=Workload(10, 10, 10, 2, "0-1"),
            nominal_peak_gflops=1.0,
            fp32_flops_per_cycle=16,
            active_frequency_ghz=2.0,
            stages_ms={
                "hot_pipeline": 0.02,
                "ideal_data": 0.01,
                "private_trace": 0.03,
                "shared_trace": 0.04,
                "full_driver": 0.10,
            },
            auxiliary_ms=0.005,
            scheduling_ms=0.005,
            driver_control_ms=0.005,
        )
        self.assertAlmostEqual(got["closure_ms"], 0.10)
        by_name = {x["name"]: x for x in got["components"]}
        self.assertLess(by_name["load_store_data_supply"]["milliseconds"], 0)

    def test_atomic_publish_refuses_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            _atomic_publish(output, {"manifest.json": {"status": "ok"}})
            self.assertEqual(
                json.loads((output / "manifest.json").read_text())["status"], "ok"
            )
            with self.assertRaisesRegex(PerfDiagError, "already exists"):
                _atomic_publish(output, {"manifest.json": {"status": "new"}})
            self.assertEqual(
                json.loads((output / "manifest.json").read_text())["status"], "ok"
            )

    def test_fake_pmu_runs_events_in_separate_controlled_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "runner"
            executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            executable.chmod(0o755)
            calls = []

            def fake_run(command, **kwargs):
                calls.append(command)
                request_path = Path(command[command.index("--request") + 1])
                request = json.loads(request_path.read_text(encoding="utf-8"))
                event = command[command.index("-e") + 1]
                response = {
                    "protocol_version": 1,
                    "request_id": request["request_id"],
                    "status": "diagnosed",
                    "samples_ms": [1.0, 1.0, 1.0],
                }
                return subprocess.CompletedProcess(
                    command, 0, json.dumps(response) + "\n",
                    f"1000;;{event};100;100\n",
                )

            runner = MatOptRunner(executable)
            with mock.patch("subprocess.run", side_effect=fake_run):
                got = runner.perf_diagnose_stage(
                    Workload(2, 3, 4, 1, "0"), {"M_blk": 2}, "fp",
                    "full_driver", {"cycles": "cycles", "instructions": "inst"},
                )
            self.assertEqual(len(calls), 2)
            self.assertNotIn("cycles,inst", " ".join(calls[0]))
            self.assertEqual(got["pmu"]["cycles"]["value"], 1000)
            self.assertEqual(got["pmu"]["instructions"]["value"], 1000)


if __name__ == "__main__":
    unittest.main()
