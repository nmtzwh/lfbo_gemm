import stat
import tempfile
import unittest
from pathlib import Path

from matopt.protocol import Workload
from matopt.runner import MatOptRunner


class RunnerTests(unittest.TestCase):
    def script(self, directory, body):
        path = Path(directory) / "runner"
        path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
        return path

    def test_protocol_error_is_classified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.script(directory, 'printf "not-json\\n"\n')
            result = MatOptRunner(path).capabilities(Workload(1, 1, 1, 1, "0"))
            self.assertEqual(result["status"], "protocol_error")

    def test_signal_is_classified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self.script(directory, "kill -SEGV $$\n")
            result = MatOptRunner(path).capabilities(Workload(1, 1, 1, 1, "0"))
            self.assertEqual(result["status"], "crashed")
            self.assertEqual(result["signal_name"], "SIGSEGV")


if __name__ == "__main__":
    unittest.main()
