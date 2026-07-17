import tempfile
import unittest
from pathlib import Path

from matopt.protocol import MeasurementProfile, Workload
from matopt.search.lfbo import LFBOConfig
from matopt.session import TuningSession
from matopt.space_config import SpaceConfig


PLAN = {
    "schema_version": 1,
    "M_blk": 64,
    "N_blk": 64,
    "K_blk": 64,
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


def response(plan, latency):
    return {
        "protocol_version": 1,
        "request_id": "fake",
        "status": "benchmarked",
        "fingerprint": "fp",
        "effective_plan": plan,
        "measurement": {
            "one_shot_ms": latency,
            "steady_ms": latency,
            "throughput_gflops": 1 / latency,
            "median_ms": latency,
            "minimum_ms": latency,
            "p90_ms": latency,
            "scratchpad_bytes": 0,
            "correct": True,
            "stable": True,
        },
    }


class FakeRunner:
    def __init__(self):
        self.evaluations = 0

    def capabilities(self, workload):
        return {
            "status": "capabilities",
            "fingerprint": "fp",
            "constraints": {"allow_split_k": True},
            "domains": {
                "pack_a": ["direct"],
                "pack_b": ["direct"],
                "bd_block": [0, 4],
                "ld_block2": [0, 2],
                "loop_order": ["default", "one-load"],
            },
        }

    def baseline(self, workload, measurement, fingerprint):
        return response(PLAN, 10)

    def evaluate(self, workload, plan, measurement, fingerprint):
        self.evaluations += 1
        return response(plan, 8)


class OnePlanSearch:
    def __init__(self, plan):
        self.plan = plan
        self.used = False

    def ask(self):
        if self.used:
            raise StopIteration
        self.used = True
        return self.plan

    def tell(self, plan, result):
        pass


class SeedRecordingSearch(OnePlanSearch):
    def __init__(self, plan):
        super().__init__(plan)
        self.seeds = []

    def seed(self, plan, result):
        self.seeds.append(plan)


class SessionTests(unittest.TestCase):
    def test_strict_space_keeps_baseline_for_comparison_only(self):
        class SlowCandidateRunner(FakeRunner):
            def evaluate(self, workload, plan, measurement, fingerprint):
                self.evaluations += 1
                return response(plan, 12)

        with tempfile.TemporaryDirectory() as directory:
            plan = dict(PLAN, loop_order="one-load")
            search = SeedRecordingSearch(plan)
            result = TuningSession(
                Workload(64, 64, 64, 1, "0"),
                SlowCandidateRunner(),
                history=Path(directory) / "history.jsonl",
            ).tune(
                budget=1,
                search=search,
                space_config=SpaceConfig.from_dict(
                    {"domains": {"loop_order": ["one-load"]}}
                ),
                measurement=MeasurementProfile(samples=1, minimum_sample_ms=1),
            )
            self.assertEqual(result["baseline"]["plan"]["loop_order"], "default")
            self.assertEqual(
                result["selected"]["one_shot"]["plan"]["loop_order"],
                "one-load",
            )
            self.assertTrue(
                all(item["plan"]["loop_order"] == "one-load" for item in result["pareto"])
            )
            self.assertEqual(search.seeds, [])

    def test_progress_events_cover_runtime_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            events = []
            TuningSession(
                Workload(64, 64, 64, 1, "0"),
                FakeRunner(),
                history=Path(directory) / "history.jsonl",
            ).tune(
                budget=1,
                search=OnePlanSearch(dict(PLAN, N_chunk_size=2)),
                progress=lambda event, payload: events.append(event),
                measurement=MeasurementProfile(samples=1, minimum_sample_ms=1),
            )
            self.assertEqual(events, ["start", "baseline", "candidate", "finish"])

    def test_resume_does_not_repeat_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.jsonl"
            plan = dict(PLAN, N_chunk_size=2)
            runner = FakeRunner()
            session = TuningSession(
                Workload(64, 64, 64, 1, "0"), runner, history=history
            )
            first = session.tune(
                budget=1,
                search=OnePlanSearch(plan),
                measurement=MeasurementProfile(samples=1, minimum_sample_ms=1),
            )
            self.assertEqual(runner.evaluations, 1)
            self.assertEqual(first["selected"]["one_shot"]["plan"], plan)
            resumed = session.tune(
                budget=1,
                search=OnePlanSearch(plan),
                measurement=MeasurementProfile(samples=1, minimum_sample_ms=1),
            )
            self.assertEqual(runner.evaluations, 1)
            self.assertEqual(resumed["selected"]["one_shot"]["plan"], plan)

    def test_builtin_lfbo_session_records_search_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.jsonl"
            runner = FakeRunner()
            result = TuningSession(
                Workload(256, 256, 256, 1, "0"), runner, history=history
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
                measurement=MeasurementProfile(samples=1, minimum_sample_ms=1),
            )
            self.assertEqual(result["search"]["algorithm"], "lfbo_pattern_search")
            terminal = [
                __import__("json").loads(line)
                for line in history.read_text(encoding="utf-8").splitlines()
                if '"phase":"search"' in line and '"state":"started"' not in line
            ]
            self.assertTrue(terminal)
            self.assertTrue(
                all(
                    record["search"]["algorithm"] == "lfbo_pattern_search"
                    for record in terminal
                )
            )

    def test_space_config_is_persisted_and_changes_resume_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.jsonl"
            runner = FakeRunner()
            session = TuningSession(
                Workload(256, 256, 256, 1, "0"), runner, history=history
            )
            first_config = SpaceConfig.from_dict(
                {"inherit_baseline": False, "domains": {"M_blk": [64, 128]}}
            )
            result = session.tune(
                budget=1,
                algorithm="random",
                space_config=first_config,
                measurement=MeasurementProfile(samples=1, minimum_sample_ms=1),
            )
            self.assertNotEqual(result["fingerprint"], result["runner_fingerprint"])
            self.assertEqual(
                result["space"]["requested"], first_config.to_dict()
            )
            second_config = SpaceConfig.from_dict(
                {"inherit_baseline": False, "domains": {"M_blk": [64, 192]}}
            )
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                session.tune(
                    budget=1,
                    algorithm="random",
                    space_config=second_config,
                    measurement=MeasurementProfile(samples=1, minimum_sample_ms=1),
                )


if __name__ == "__main__":
    unittest.main()
