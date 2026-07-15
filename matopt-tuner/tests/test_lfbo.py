import math
import unittest

from matopt.protocol import Workload, stable_hash
from matopt.search.lfbo import LFBOConfig, LFBOSearch
from matopt.space import PlanSpace


BASELINE = {
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

CAPABILITIES = {
    "constraints": {"allow_split_k": True},
    "domains": {
        "pack_a": ["direct", "per_call_padded"],
        "pack_b": ["direct", "per_call_n32", "per_call_n64"],
        "bd_block": [0, 4, 5, 6, 7],
        "ld_block2": [0, 2, 3, 4],
        "loop_order": ["default", "one-load"],
    },
}


def benchmarked(latency):
    return {
        "status": "benchmarked",
        "measurement": {
            "correct": True,
            "one_shot_ms": latency,
            "steady_ms": latency,
            "throughput_gflops": 1000 / latency,
        },
    }


class LFBOTests(unittest.TestCase):
    def setUp(self):
        self.space = PlanSpace(
            Workload(4096, 4096, 4096, 4, "0-3"), BASELINE, CAPABILITIES
        )

    def test_encoding_and_multifield_neighbors(self):
        plans = self.space.random_plans(8, __import__("random").Random(7))
        encoded = self.space.encode(plans)
        self.assertEqual(encoded.shape[0], 8)
        self.assertGreater(encoded.shape[1], len(self.space.fields))
        neighbors = self.space.neighbors(
            BASELINE, 20, 2, __import__("random").Random(11)
        )
        self.assertEqual(len({stable_hash(plan) for plan in neighbors}), len(neighbors))
        self.assertTrue(
            any(
                sum(plan[k] != BASELINE[k] for k in self.space.fields) > 1
                for plan in neighbors
            )
        )

    def test_classifier_learns_from_latency_and_failures(self):
        search = LFBOSearch(
            self.space,
            objective="one_shot",
            budget=10,
            seed=17,
            config=LFBOConfig(
                initial_population=4,
                copies=2,
                generations=3,
                neighbors=24,
                radius=2,
                quantile=0.25,
                fraction_selected=0.25,
                patience=3,
                trees=16,
            ),
        )
        search.seed(BASELINE, benchmarked(10.0))
        proposed = []
        while True:
            try:
                plan = search.ask()
            except StopIteration:
                break
            proposed.append(plan)
            if len(proposed) == 1:
                search.tell(plan, {"status": "rejected"})
            else:
                latency = 20.0 - plan["M_blk"] / 32.0
                search.tell(plan, benchmarked(latency))
        self.assertLessEqual(len(proposed), 10)
        self.assertEqual(len({stable_hash(plan) for plan in proposed}), len(proposed))
        self.assertTrue(any(math.isinf(item.loss) for item in search.observations))
        self.assertIsNotNone(search.model)

    def test_resume_restores_total_budget(self):
        config = LFBOConfig(
            initial_population=2,
            copies=1,
            generations=2,
            neighbors=8,
            fraction_selected=0.5,
            patience=2,
            trees=8,
        )
        search = LFBOSearch(
            self.space, objective="one_shot", budget=3, seed=3, config=config
        )
        search.seed(BASELINE, benchmarked(10))
        search.restore_budget_used(2)
        plan = search.ask()
        search.tell(plan, benchmarked(9))
        with self.assertRaises(StopIteration):
            search.ask()


if __name__ == "__main__":
    unittest.main()
