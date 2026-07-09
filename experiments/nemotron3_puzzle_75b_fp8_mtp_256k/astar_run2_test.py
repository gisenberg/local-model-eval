from __future__ import annotations

import heapq
from typing import List, Tuple, Optional, Dict


class AStarGrid:
    """
    A* path‑finder on a rectangular 2‑D grid.

    Parameters
    ----------
    grid : List[List[int]]
        The navigation map.  A cell value of ``0`` is treated as an
        impassable wall.  Positive integers are traversable and their
        value is the **cost to enter** that cell (the cost of the start
        cell itself is not counted).

    The algorithm uses 4‑directional movement (up, down, left, right)
    and the Manhattan heuristic, which is admissible for non‑negative
    integer costs.

    Methods
    -------
    find_path(start, end)
        Returns the optimal path as a list of ``(row, col)`` tuples
        (including ``start`` and ``end``) or ``None`` if no path exists.
        Raises ``ValueError`` if ``start`` or ``end`` are out of bounds
        or lie on a wall.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        if not grid or not grid[0]:
            raise ValueError("grid must be non‑empty")
        self._grid = grid
        self._rows = len(grid)
        self._cols = len(grid[0])

        # sanity check: all rows must have the same length
        if any(len(row) != self._cols for row in grid):
            raise ValueError("all rows in the grid must have the same length")

    # -----------------------------------------------------------------
    # internal helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance – admissible for 4‑directional grid."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self._rows and 0 <= c < self._cols

    def _is_passable(self, r: int, c: int) -> bool:
        """A cell is passable if it is inside the grid and not a wall."""
        return self._in_bounds(r, c) and self._grid[r][c] > 0

    def _neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """4‑directional neighbours."""
        r, c = pos
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [n for n in candidates if self._is_passable(*n)]

    # -----------------------------------------------------------------
    # public API
    # -----------------------------------------------------------------
    def find_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find the cheapest path from *start* to *end* using A*.

        Returns
        -------
        Optional[List[Tuple[int, int]]]
            List of coordinates from start to end (inclusive) if a path
            exists; otherwise ``None``.

        Raises
        ------
        ValueError
            If ``start`` or ``end`` are out of bounds or located on a wall.
        """
        # ----- validation -------------------------------------------------
        if not self._in_bounds(*start) or not self._in_bounds(*end):
            raise ValueError("start or end is out of grid bounds")
        if not self._is_passable(*start) or not self._is_passable(*end):
            raise ValueError("start or end is located on a wall (value 0)")

        if start == end:
            return [start]

        # ----- A* core ----------------------------------------------------
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []  # (f, g, node)
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))

        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == end:
                # reconstruct path
                path: List[Tuple[int, int]] = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for neighbor in self._neighbors(current):
                tentative_g = current_g + self._grid[neighbor[0]][neighbor[1]]
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f, tentative_g, neighbor))

        # open set exhausted → no path
        return None

import pytest
from typing import List, Tuple

# Assuming the implementation above is saved in a file named `astar_grid.py`


def test_simple_unweighted_grid():
    """All cells cost 1 – the shortest Manhattan path should be found."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Manhattan distance is 4 steps → 5 cells in the path
    assert len(path) == 5
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # verify each step is orthogonal
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1


def test_start_equals_end():
    """When start == end the function must return a single‑element list."""
    grid = [[1, 2], [3, 4]]
    astar = AStarGrid(grid)
    assert astar.find_path((1, 0), (1, 0)) == [(1, 0)]


def test_wall_blocks_path():
    """A full column of zeros should make the target unreachable."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None


def test_out_of_bounds_raises():
    """Accessing coordinates outside the grid must raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 0))


def test_weighted_prefers_low_cost_route():
    """
    The algorithm must choose a lower‑total‑cost path even if it is longer
    in terms of step count.
    """
    grid = [
        [1, 100, 1],
        [1, 100, 1],
        [1,   1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Expected optimal route: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)
    expected = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert path == expected
    # Compute total cost (excluding the start cell)
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4  # 1+1+1+1


def test_path_contains_only_orthogonal_moves():
    """Ensure the returned path never contains a diagonal step."""
    grid = [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
    ]
    astar = AStarGrid(grid)
    # Use a case where many equivalent shortest paths exist;
    # we just check that whatever path is returned obeys 4‑directionality.
    path = astar.find_path((0, 0), (3, 3))
    assert path is not None
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        assert abs(r1 - r2) + abs(c1 - c2) == 1  # Manhattan distance == 1


if __name__ == "__main__":
    # Allow running the file directly for a quick sanity check
    pytest.main([__file__])