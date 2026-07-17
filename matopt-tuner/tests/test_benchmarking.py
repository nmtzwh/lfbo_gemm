import unittest

from matopt.benchmarking import BenchmarkError, make_summary, percentile


class BenchmarkSummaryTests(unittest.TestCase):
    def test_nearest_rank(self):
        self.assertEqual(percentile(range(1, 11), 0.9), 9)

    def test_summary_ratios(self):
        manifest = {
            "kernel_id": "kid",
            "objective": "steady",
            "selected_measurement": {"steady_ms": 2},
            "workload": {"m": 10, "n": 20, "k": 30},
        }
        cases = {
            "MatOpt/create": [1, 1, 1],
            "MatOpt/prepare_weights": [2, 2, 2],
            "MatOpt/one_shot": [4, 4, 4],
            "MatOpt/steady_throughput": [3, 3, 3],
            "OpenBLAS/sgemm": [6, 6, 6],
        }
        summary = make_summary(manifest, cases, affinity="0", openblas={"interface": "ILP64"})
        self.assertEqual(summary["kernel_id"], "kid")
        self.assertEqual(summary["ratios"]["one_shot_latency_vs_openblas"], 2 / 3)
        self.assertEqual(summary["ratios"]["steady_gflops_vs_openblas"], 2)

    def test_missing_case_fails(self):
        with self.assertRaises(BenchmarkError):
            make_summary({"workload": {"m": 1, "n": 1, "k": 1}}, {}, affinity="0", openblas={})


if __name__ == "__main__":
    unittest.main()
