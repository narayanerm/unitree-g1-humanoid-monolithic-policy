##########################################################################################
# Project    : unitree-g1-humanoid-monolithic-policy                                     #
# Developer  : Narayane Ribeiro Medeiros                                                 #
# Repository : https://github.com/narayanerm/unitree-g1-humanoid-monolithic-policy       #
##########################################################################################
"""8-connected A* on a 2D occupancy grid for the walk-phase approach.

Ported from the earlier prototype in
``../Humanoid-Loco-Manipulation/notebooks/04_state_machine.ipynb`` (functions
``_astar`` + ``plan_astar``). The current FSM in ``fsm_continuous_batch`` uses
reactive pure-pursuit straight at the box; this planner lets the dataset
builder produce demonstrations that first detour around the pick table before
handing over to the pick expert, so the monolithic student sees "go around
the obstacle then grab" trajectories instead of straight-line pulls.

Only axis-aligned rectangular obstacles are supported. Obstacles are inflated
by the robot footprint ``robot_r`` before rasterisation. Start and goal cells
that fall inside an inflated obstacle are BFS-snapped to the nearest free cell,
so a robot spawn at 0.2 m from the table (well within the inflated footprint)
still returns a valid path.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


# Pick-table footprint (cx, cy, hx, hy) in world frame, matching ``policy_pick.TABLE_X`` (0.60 m); exported so ``fsm_continuous_batch`` imports one canonical constant.
PICK_TABLE_OBSTACLE = (0.60, 0.0, 0.30, 0.40)


@dataclass
class AstarConfig:
    """Grid config for the 2D planner."""
    arena_half: float = 6.0
    res: float = 0.15
    robot_r: float = 0.35
    arrival_dist: float = 0.30


def _build_occupancy(cfg: AstarConfig, obstacles: list[tuple[float, float, float, float]]) -> np.ndarray:
    """Rasterise inflated rectangular obstacles into a boolean grid.
    Args: cfg: grid config; obstacles: list of (cx, cy, half_x, half_y) world-frame boxes.
    Returns: (n, n) bool grid, True where blocked (obstacle inflated by ``cfg.robot_r``).
    """
    n = int(2 * cfg.arena_half / cfg.res) + 1
    xs = np.linspace(-cfg.arena_half, cfg.arena_half, n)
    ys = np.linspace(-cfg.arena_half, cfg.arena_half, n)
    occ = np.zeros((n, n), dtype=bool)
    for cx, cy, hx, hy in obstacles:
        inflated_hx = hx + cfg.robot_r
        inflated_hy = hy + cfg.robot_r
        i_mask = np.abs(xs - cx) <= inflated_hx
        j_mask = np.abs(ys - cy) <= inflated_hy
        occ[np.ix_(i_mask, j_mask)] = True
    return occ


def _world_to_cell(xy, cfg: AstarConfig) -> tuple[int, int]:
    """Map world xy to grid indices with origin at the arena corner.
    Args: xy: (x, y) world coordinates; cfg: grid config supplying ``arena_half`` and ``res``.
    Returns: (i, j) integer cell indices with (0, 0) at (-arena_half, -arena_half).
    """
    i = int(round((float(xy[0]) + cfg.arena_half) / cfg.res))
    j = int(round((float(xy[1]) + cfg.arena_half) / cfg.res))
    return i, j


def _cell_to_world(ij: tuple[int, int], cfg: AstarConfig) -> tuple[float, float]:
    """Map grid indices back to world xy at the cell centre.
    Args: ij: (i, j) integer cell indices; cfg: grid config supplying ``arena_half`` and ``res``.
    Returns: (x, y) world coordinates of the cell centre.
    """
    x = -cfg.arena_half + ij[0] * cfg.res
    y = -cfg.arena_half + ij[1] * cfg.res
    return x, y


def _snap_to_free(cell: tuple[int, int], occ: np.ndarray) -> Optional[tuple[int, int]]:
    """BFS-snap a grid cell to the nearest free cell under 4-connected moves.
    Args: cell: (i, j) starting cell; occ: (n, n) bool grid where True marks blocked cells.
    Returns: (i, j) of the nearest free cell, or None if ``cell`` is out of bounds.
    """
    n0, n1 = occ.shape
    if not (0 <= cell[0] < n0 and 0 <= cell[1] < n1):
        return None
    if not occ[cell]:
        return cell
    seen = {cell}
    q = deque([cell])
    while q:
        c = q.popleft()
        if not occ[c]:
            return c
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = c[0] + di, c[1] + dj
            if 0 <= ni < n0 and 0 <= nj < n1 and (ni, nj) not in seen:
                seen.add((ni, nj))
                q.append((ni, nj))
    return None


_NEIGHBOURS = (
    (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
    (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142),
)


def _astar(occ: np.ndarray, s: tuple[int, int], t: tuple[int, int]) -> Optional[list[tuple[int, int]]]:
    """Run 8-connected A* with the octile heuristic over an occupancy grid.
    Args: occ: (n, n) bool grid, True where blocked; s: start cell (i, j); t: goal cell (i, j).
    Returns: list of cells from ``s`` to ``t`` inclusive, or None if ``t`` is unreachable.
    """
    n0, n1 = occ.shape

    def h(c):
        """Octile distance from ``c`` to ``t``, admissible for 8-connected unit-cost moves."""
        dx = abs(c[0] - t[0])
        dy = abs(c[1] - t[1])
        return (dx + dy) + (1.4142 - 2.0) * min(dx, dy)

    open_q: list[tuple[float, tuple[int, int]]] = [(h(s), s)]
    came: dict[tuple[int, int], Optional[tuple[int, int]]] = {s: None}
    g_score: dict[tuple[int, int], float] = {s: 0.0}
    while open_q:
        _, c = heapq.heappop(open_q)
        if c == t:
            path = [c]
            while came[path[-1]] is not None:
                path.append(came[path[-1]])
            return list(reversed(path))
        gc = g_score[c]
        for di, dj, cost in _NEIGHBOURS:
            ni, nj = c[0] + di, c[1] + dj
            if not (0 <= ni < n0 and 0 <= nj < n1) or occ[ni, nj]:
                continue
            ng = gc + cost
            nb = (ni, nj)
            if ng < g_score.get(nb, float("inf")):
                g_score[nb] = ng
                came[nb] = c
                heapq.heappush(open_q, (ng + h(nb), nb))
    return None


def plan_path(
    start_xy,
    goal_xy,
    obstacles: list[tuple[float, float, float, float]],
    cfg: Optional[AstarConfig] = None,
) -> Optional[np.ndarray]:
    """Plan an 8-connected A* path from ``start_xy`` to ``goal_xy`` around inflated obstacles.
    Args: start_xy: (x, y) world spawn; goal_xy: (x, y) world target (snapped to the nearest
    free cell if it falls inside an inflated obstacle); obstacles: list of
    (cx, cy, half_x, half_y) rectangles; cfg: grid config, defaults to ``AstarConfig()``.
    Returns: (N, 2) float32 array of world-frame waypoints, or None if unreachable.
    """
    cfg = cfg or AstarConfig()
    occ = _build_occupancy(cfg, obstacles)
    s = _snap_to_free(_world_to_cell(start_xy, cfg), occ)
    t = _snap_to_free(_world_to_cell(goal_xy, cfg), occ)
    if s is None or t is None:
        return None
    cells = _astar(occ, s, t)
    if cells is None:
        return None
    return np.array([_cell_to_world(c, cfg) for c in cells], dtype=np.float32)


class WaypointTracker:
    """Stateful lookahead over an A* path: advance the active index as the robot arrives."""

    def __init__(self, waypoints: np.ndarray, arrival_dist: float = 0.30):
        """Store waypoints and the per-waypoint arrival radius.
        Args: waypoints: (N, 2) world-frame path; arrival_dist: radius (m) at which the
        tracker advances to the next waypoint.
        Returns: none.
        """
        self.waypoints = np.asarray(waypoints, dtype=np.float32)
        self.arrival_dist = float(arrival_dist)
        self.idx = 0

    def current(self, robot_xy: np.ndarray) -> np.ndarray:
        """Advance the index past any waypoint the robot has already reached and return the active one.
        Args: robot_xy: (2,) robot planar position in world frame.
        Returns: (2,) world-frame target for pure-pursuit.
        """
        last = len(self.waypoints) - 1
        while self.idx < last:
            d = float(np.linalg.norm(robot_xy - self.waypoints[self.idx]))
            if d < self.arrival_dist:
                self.idx += 1
            else:
                break
        return self.waypoints[self.idx]

    @property
    def done(self) -> bool:
        """True once the tracker is holding the last waypoint."""
        return self.idx >= len(self.waypoints) - 1
