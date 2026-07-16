import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from matopt.visualization import (
    load_records,
    pareto_steps,
    plot_trajectory,
    trajectory_points,
)


def record(timestamp, latency, generation=None, phase="search"):
    value = {
        "fingerprint": "fp",
        "timestamp_ns": timestamp,
        "state": "benchmarked",
        "phase": phase,
        "plan_hash": str(timestamp),
        "response": {
            "measurement": {
                "correct": True,
                "one_shot_ms": latency,
                "steady_ms": latency + 1,
                "median_ms": latency + 2,
            }
        },
    }
    if generation is not None:
        value["search"] = {"generation": generation}
    return value


class VisualizationTests(unittest.TestCase):
    def test_points_and_generation_pareto_steps(self):
        records = [
            record(1, 10, phase="baseline"),
            record(2, 12, generation=0),
            record(3, 8, generation=0),
            record(4, 9, generation=1),
            record(5, 7, generation=1),
            {"fingerprint": "fp", "timestamp_ns": 6, "state": "rejected"},
        ]
        points = trajectory_points(records)
        self.assertEqual([point.generation for point in points], [-1, 0, 0, 1, 1])
        self.assertEqual(
            pareto_steps(points),
            [(0, -1, 10.0), (2, 0, 8.0), (4, 1, 7.0)],
        )

    def test_load_tolerates_truncated_tail_and_rejects_mixed_history(self):
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.jsonl"
            history.write_text(
                json.dumps(record(1, 10, phase="baseline")) + "\n{",
                encoding="utf-8",
            )
            self.assertEqual(len(load_records(history)), 1)
            history.write_text(
                json.dumps(record(1, 10))
                + "\n"
                + json.dumps(dict(record(2, 9), fingerprint="other"))
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "multiple fingerprints"):
                load_records(history)

    def test_renders_png(self):
        if importlib.util.find_spec("matplotlib") is None:
            self.skipTest("matplotlib is not installed")
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.jsonl"
            output = Path(directory) / "trajectory.png"
            history.write_text(
                "\n".join(
                    json.dumps(item)
                    for item in [
                        record(1, 10, phase="baseline"),
                        record(2, 9, generation=0),
                        record(3, 8, generation=1),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            plot_trajectory(history, output)
            self.assertGreater(output.stat().st_size, 1000)
            self.assertEqual(output.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
