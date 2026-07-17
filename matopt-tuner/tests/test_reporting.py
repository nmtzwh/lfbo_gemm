import io
import unittest

from matopt.protocol import Workload
from matopt.reporting import ConsoleReporter


PLAN = {
    "M_blk": 64,
    "N_blk": 64,
    "K_blk": 256,
    "M_chunk_size": 1,
    "N_chunk_size": 1,
    "brgemm_batch_size": 1,
    "nthr_k": 1,
    "pack_a": "direct",
    "pack_b": "direct",
    "bd_block": 0,
    "ld_block2": 0,
    "loop_order": "default",
}


def record(latency, generation=None, state="benchmarked", plan_hash="p"):
    value = {
        "state": state,
        "plan": dict(PLAN),
        "plan_hash": plan_hash,
        "response": {},
    }
    if state == "benchmarked":
        value["response"]["measurement"] = {
            "one_shot_ms": latency,
            "steady_ms": latency * 0.9,
            "throughput_gflops": 1000 / latency,
            "correct": True,
            "stable": True,
        }
    else:
        value["response"]["reason_code"] = "invalid_plan"
    if generation is not None:
        value["search"] = {"generation": generation}
    return value


def start_payload():
    return {
        "workload": Workload(256, 256, 256, 4, "0-3"),
        "capabilities": {"effective_isa": "AVX2"},
        "space": {
            "effective_domains": {
                "M_blk": [64, 128],
                "N_blk": [32, 64],
                "K_blk": [256],
                "M_chunk_size": [1, 2],
                "N_chunk_size": [1, 2, 4],
                "brgemm_batch_size": [1, 2],
                "nthr_k": [1, 2],
                "pack_a": ["direct", "per_call_padded"],
                "pack_b": ["direct", "persistent_n64"],
                "bd_block": [0, 4],
                "ld_block2": [0, 2],
                "loop_order": ["default", "one-load"],
            },
            "limits": {
                "scratchpad_per_thread_bytes": 64 * 1024 * 1024,
                "minimum_parallel_work_per_thread": 0.5,
            },
        },
        "objective": "one_shot",
        "algorithm": "lfbo",
        "budget": 8,
        "resumed": 0,
    }


class ReportingTests(unittest.TestCase):
    def test_generation_summary_and_selection_are_compact(self):
        stream = io.StringIO()
        reporter = ConsoleReporter(level=1, stream=stream, color="never")
        baseline = record(10, plan_hash="base")
        best = record(8, generation=0, plan_hash="best")
        rejected = record(0, generation=1, state="rejected", plan_hash="bad")
        reporter("start", start_payload())
        reporter("baseline", {"record": baseline})
        reporter("candidate", {"record": best})
        reporter("candidate", {"record": rejected})
        reporter(
            "finish",
            {
                "result": {
                    "selected": {"one_shot": best},
                    "pareto": [best],
                }
            },
        )
        output = stream.getvalue()
        self.assertIn("MATOPT 256x256x256", output)
        self.assertIn("SPACE blocks", output)
        self.assertIn("scratch=64MiB/thr work/thr>=0.5", output)
        self.assertIn("GEN 0 ok=1 rejected=0", output)
        self.assertIn("GEN 1 ok=0 rejected=1", output)
        self.assertIn("SELECT one_shot 8 ms gain=+20.00%", output)
        self.assertNotIn("#001", output)
        self.assertNotIn("\033[", output)

    def test_detailed_mode_and_forced_color(self):
        stream = io.StringIO()
        reporter = ConsoleReporter(level=2, stream=stream, color="always")
        baseline = record(10, plan_hash="base")
        reporter("start", start_payload())
        reporter("baseline", {"record": baseline})
        reporter("candidate", {"record": record(9, generation=0)})
        reporter(
            "finish",
            {
                "result": {
                    "selected": {"one_shot": record(9, generation=0)},
                    "pareto": [],
                }
            },
        )
        output = stream.getvalue()
        self.assertIn("#001 g0", output)
        self.assertIn("OK", output)
        self.assertIn("\033[", output)

    def test_resumed_records_initialize_current_best(self):
        stream = io.StringIO()
        reporter = ConsoleReporter(level=1, stream=stream, color="never")
        baseline = record(10, plan_hash="base")
        previous = record(7, generation=0, plan_hash="old")
        reporter("start", dict(start_payload(), resumed=1))
        reporter(
            "baseline", {"record": baseline, "prior_records": [previous]}
        )
        reporter(
            "finish",
            {
                "result": {
                    "selected": {"one_shot": previous},
                    "pareto": [previous],
                }
            },
        )
        self.assertIn("resumed=1", stream.getvalue())
        self.assertIn("SELECT one_shot 7 ms gain=+30.00%", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
