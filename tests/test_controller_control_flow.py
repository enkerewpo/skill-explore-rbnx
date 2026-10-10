import asyncio
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from explore_skill.controller import ExploreController, TaskHandle


def _map_msg():
    """OccupancyGrid-like message: free 10 m x 10 m room with two
    unknown pockets, i.e. two separate frontier clusters."""
    data = np.zeros((100, 100), dtype=np.int8)
    data[20:30, 60:70] = -1
    data[70:80, 60:70] = -1
    origin = SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0))
    info = SimpleNamespace(width=100, height=100, resolution=0.1,
                           origin=origin)
    return SimpleNamespace(info=info, data=data.tobytes())


class _Grid:
    @staticmethod
    def world_to_cell(_x, _y):
        return (4, 7)


class ExploreControlFlowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.controller = ExploreController(
            map_topic="/map",
            nav_navigate_endpoint="http://127.0.0.1:1/mcp/",
            nav_status_endpoint="http://127.0.0.1:1/mcp/",
            nav_cancel_endpoint="http://127.0.0.1:1/mcp/",
        )
        self.controller._latest_pose_xyyaw = (1.0, 2.0, 0.0)
        self.controller._latest_map = object()

    def _handle(self, legs: int = 0) -> TaskHandle:
        return TaskHandle(
            task_id="test", started_at=time.time(), timeout_s=60.0,
            max_speed_m_s=0.1, legs_completed=legs,
        )

    @patch("explore_skill.frontier.GridView.from_msg", return_value=_Grid())
    def test_sweep_requires_periodic_leg_and_missing_coverage(self, _grid) -> None:
        self.controller._viewed_sectors[(4, 7)] = set()
        self.assertFalse(self.controller._should_sweep(self._handle(1)))
        self.assertFalse(self.controller._should_sweep(self._handle(2)))
        self.assertTrue(self.controller._should_sweep(self._handle(3)))

        self.controller._viewed_sectors[(4, 7)] = set(range(6))
        self.assertFalse(self.controller._should_sweep(self._handle(3)))

    def test_navigation_timeout_cancels_the_accepted_run(self) -> None:
        calls = []

        def rpc(tool, args):
            calls.append((tool, args))
            if tool == "navigate":
                return {"accepted": True, "run_id": "run-42"}
            if tool == "status":
                return {"state": "RUNNING"}
            return {"accepted": True}

        self.controller._mcp_call_sync = rpc
        self.controller.NAV_POLL_PERIOD_S = 0.001
        ok, detail = self.controller._nav_navigate_blocking(
            1.0, 2.0, yaw=None, timeout_s=0.003,
            cancel_evt=self._handle(),
        )

        self.assertFalse(ok)
        self.assertEqual(detail, "leg timeout")
        self.assertIn(("cancel", {"run_id": "run-42"}), calls)

    def test_navigation_cancel_request_cancels_the_accepted_run(self) -> None:
        calls = []

        def rpc(tool, args):
            calls.append((tool, args))
            if tool == "navigate":
                return {"accepted": True, "run_id": "run-7"}
            return {"state": "RUNNING"}

        handle = self._handle()
        handle.cancel_requested = True
        self.controller._mcp_call_sync = rpc
        ok, detail = self.controller._nav_navigate_blocking(
            1.0, 2.0, yaw=None, timeout_s=1.0, cancel_evt=handle,
        )

        self.assertFalse(ok)
        self.assertEqual(detail, "canceled during nav")
        self.assertIn(("cancel", {"run_id": "run-7"}), calls)


    def _run_with_nav(self, nav):
        self.controller._latest_map = _map_msg()
        self.controller._latest_pose_xyyaw = (5.0, 5.0, 0.0)
        self.controller.LOOP_QUIET_PERIOD_S = 0.0
        self.controller._nav_navigate_blocking = nav
        handle = self._handle()
        self.controller._task = handle
        self.controller._run_task(handle)
        return handle

    def test_failed_frontier_is_not_retried(self) -> None:
        goals = []

        def nav(x, y, *, yaw, timeout_s, cancel_evt):
            goals.append((x, y, yaw))
            return False, "nav terminal: FAILED"

        handle = self._run_with_nav(nav)

        legs = [(x, y) for x, y, yaw in goals if yaw is None]
        self.assertEqual(len(legs), 2)
        (ax, ay), (bx, by) = legs
        self.assertGreater(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5, 1.0)
        # Both frontiers failed: a bounded number of sweeps, then give up
        # instead of looping until the global timeout.
        sweeps = [g for g in goals if g[2] is not None]
        self.assertLessEqual(len(sweeps),
                             4 * self.controller.MAX_EMPTY_SWEEPS)
        self.assertEqual(handle.state, "error")
        self.assertIn("no reachable frontier", handle.detail)

    def test_consecutive_nav_failures_end_the_task(self) -> None:
        legs = []

        def nav(x, y, *, yaw, timeout_s, cancel_evt):
            # Every frontier is unreachable, and there are always new
            # ones: forget earlier failures so a fresh target exists.
            cancel_evt.failed_targets.clear()
            if yaw is None:
                legs.append((x, y))
            return False, "nav terminal: FAILED"

        handle = self._run_with_nav(nav)

        self.assertEqual(len(legs),
                         self.controller.MAX_CONSECUTIVE_NAV_FAILURES)
        self.assertEqual(handle.state, "error")
        self.assertIn("nav legs failed in a row", handle.detail)

    def test_mcp_call_works_inside_a_running_event_loop(self) -> None:
        async def fake_call(tool, args):
            return {"tool": tool, "args": args}

        self.controller._mcp_call = fake_call

        async def handler():
            return self.controller._mcp_call_sync("cancel", {"run_id": ""})

        self.assertEqual(asyncio.run(handler()),
                         {"tool": "cancel", "args": {"run_id": ""}})


if __name__ == "__main__":
    unittest.main()
