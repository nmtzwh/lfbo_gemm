import tempfile
import unittest
from pathlib import Path

from matopt.history import History
from matopt.objectives import pareto, select
from matopt.protocol import Workload, canonical_json, stable_hash
from matopt.space import candidates


def measurement(one_shot, steady, throughput, scratch=0):
    return {
        "one_shot_ms": one_shot,
        "steady_ms": steady,
        "throughput_gflops": throughput,
        "median_ms": one_shot,
        "minimum_ms": one_shot,
        "p90_ms": one_shot,
        "scratchpad_bytes": scratch,
        "correct": True,
        "stable": True,
    }


def record(plan, value):
    return {
        "state": "benchmarked",
        "plan": plan,
        "response": {"measurement": value},
    }


class CoreTests(unittest.TestCase):
    def test_canonical_hash(self):
        self.assertEqual(canonical_json({"b": 2, "a": 1}), '{"a":1,"b":2}')
        self.assertEqual(stable_hash({"a": 1, "b": 2}), stable_hash({"b": 2, "a": 1}))

    def test_history_recovers_only_truncated_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            history = History(path, "fp")
            history.append({"fingerprint": "fp", "state": "started"})
            with path.open("ab") as stream:
                stream.write(b'{"fingerprint":')
            self.assertEqual(len(history.load()), 1)
            with path.open("ab") as stream:
                stream.write(b'}\n')
            with self.assertRaisesRegex(ValueError, "malformed complete"):
                history.load()

    def test_history_rejects_fingerprint_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            path.write_text('{"fingerprint":"old"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                History(path, "new").load()

    def test_space_obeys_split_k_capability(self):
        workload = Workload(256, 256, 256, 4, "0-3")
        baseline = {
            "schema_version": 1,
            "M_blk": 128,
            "N_blk": 64,
            "K_blk": 256,
            "M_chunk_size": 1,
            "N_chunk_size": 1,
            "K_chunk_size": 1,
            "brgemm_batch_size": 1,
            "nthr_k": 1,
            "pack_a": "direct",
            "pack_b": "direct",
            "bd_block": 0,
            "ld_block2": 0,
            "loop_order": "default",
        }
        caps = {
            "constraints": {"allow_split_k": False},
            "domains": {
                "pack_a": ["direct", "per_call_padded"],
                "pack_b": ["direct", "per_call_n32", "per_call_n64"],
                "bd_block": [0, 4],
                "ld_block2": [0, 2],
                "loop_order": ["default", "one-load"],
            },
        }
        plans = candidates(workload, baseline, caps, 50, 7)
        self.assertTrue(plans)
        self.assertTrue(all(plan["nthr_k"] == 1 for plan in plans))

    def test_pareto_and_noise_fallback(self):
        base = record({"id": 0}, measurement(10, 9, 100, 10))
        better = record({"id": 1}, measurement(8, 8, 110, 9))
        tradeoff = record({"id": 2}, measurement(7, 10, 90, 8))
        self.assertEqual(len(pareto([base, better, tradeoff])), 2)
        self.assertIs(select([base, better], "one_shot", base), better)
        noisy = record({"id": 3}, measurement(9.95, 9, 100, 10))
        self.assertIs(select([base, noisy], "one_shot", base), base)


if __name__ == "__main__":
    unittest.main()
