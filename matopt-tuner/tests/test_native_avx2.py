import os
import tempfile
import unittest
from pathlib import Path

from matopt.protocol import MeasurementProfile, Workload
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


if __name__ == "__main__":
    unittest.main()
