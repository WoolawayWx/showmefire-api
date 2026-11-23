import unittest
from unittest.mock import MagicMock

from core.loop_monitor import LoopMonitor


class LoopMonitorTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.dump = MagicMock()
        self.monitor = LoopMonitor(threshold=3.0, clock=lambda: self.now, dump=self.dump)

    def test_a_healthy_loop_is_left_alone(self):
        for _ in range(5):
            self.now += 0.5
            self.monitor.beat()
            self.assertIsNone(self.monitor.check())
        self.dump.assert_not_called()

    def test_a_blocked_loop_dumps_stacks_once_per_episode_and_reports_recovery(self):
        self.now += 3.5
        self.assertEqual(self.monitor.check(), "blocked")
        self.now += 5
        self.assertIsNone(self.monitor.check())          # same episode: no second dump
        self.assertEqual(self.dump.call_count, 1)
        self.monitor.beat()
        self.assertEqual(self.monitor.check(), "recovered")
        self.now += 4
        self.assertEqual(self.monitor.check(), "blocked")   # a new episode dumps again
        self.assertEqual(self.dump.call_count, 2)

    def test_a_failing_dump_does_not_raise(self):
        self.dump.side_effect = RuntimeError("no stderr")
        self.now += 4
        self.assertEqual(self.monitor.check(), "blocked")


if __name__ == "__main__":
    unittest.main()
