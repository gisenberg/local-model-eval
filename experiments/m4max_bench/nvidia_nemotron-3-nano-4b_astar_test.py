import pytest
from typing import List, Tuple

def test_start_equals_end():
    grid = [[0, 1, 0],
            [1, 1, 1]]
    g = AStarGrid(grid)
    start = (0, 1)
    end = (0, 1)
    assert g.find_path(start, end) == [start]

def test_simple_path():
    grid = [[0, 2, 3],
            [1, 1, 4]]
    g = AStarGrid(grid)
    start = (0, 1)
    end = (1, 2)
    # Path: down then right? Actually from (0,1) cost 2 to move down to (1,1) cost 1 total 3,
    # then right to (1,2) cost 4 total 7. Alternative: right then down? Not possible.
    path = g.find_path(start, end)
    assert path == [start, (1, 1), end]

def test_wall_blocked():
    grid = [[0, 1],
            [0, 2]]
    g = AStarGrid(grid)
    start = (0, 1)
    end = (1, 1)
    # Path exists: down cost 2 total 3
    path = g.find_path(start, end)
    assert path == [start, end]

def test_no_path():
    grid = [[0, 0],
            [0, 0]]
    g = AStarGrid(grid)
    start = (0, 1)
    end = (1, 0)
    assert g.find_path(start, end) is None

def test_out_of_bounds_error():
    grid = [[1, 2], [3, 4]]
    g = AStarGrid(grid)
    with pytest.raises(ValueError):
        g.find_path((-1, 0), (0, 0))

def test_rectangular_grid_error_init():
    grid = [[1, 2],
            [3]]  # not rectangular
    with pytest.raises(ValueError):
        AStarGrid(grid)


from __future__ import annotations

import heapq
from typing import List, Tuple, Optional


class AStarGrid:
    """
    A* path‑finder for a weighted 2‑D grid.

    * ``grid`` – list of rows; each element is an integer.
        ``0`` means the cell is impassable (a wall).
        Any positive integer is the cost to **enter** that cell.
    * Movement is limited to the four orthogonal directions
      (up, down, left, right).

    The algorithm uses a Manhattan‑distance heuristic and a
    ``heapq`` as the open set.  It returns the optimal path
    (minimum total entry‑cost) or ``None`` when no path exists.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Create the grid and validate its shape."""
        self.grid = grid
        if not grid:
            raise ValueError("grid must contain at least one row")
        self.rows = len(grid)
        self.cols = len(grid[0])
        for r in range(self.rows):
            if len(grid[r]) != self.cols:
                raise ValueError("rectangular grid required")

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def find_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Compute the optimal path from *start* to *end*.

        Parameters
        ----------
        start : (row, col)
            Coordinates of the source cell.
        end   : (row, col)
            Coordinates of the destination cell.

        Returns
        -------
        List[(row, col)] | None
            The path inclusive of both ends, or ``None`` if no route exists.

        Raises
        ------
        ValueError
            * ``start`` or ``end`` is out‑of‑bounds / a wall.
            * ``grid`` is empty or non‑rectangular (checked in ``__init__``).
        """
        # ----------------------------------------------------------------- #
        # 1️⃣  Input validation
        # ----------------------------------------------------------------- #
        self._validate_coordinates(start)
        self._validate_coordinates(end)

        if start == end:
            return [start]

        rows, cols = self.rows, self.cols

        # ----------------------------------------------------------------- #
        # 2️⃣  A* initialisation
        # ----------------------------------------------------------------- #
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        open_set: List[Tuple[int, int, int]] = []          # (f_score, r, c)
        heapq.heappush(open_set,
                        (self._heuristic(start, end) + self.grid[start[0]][start[1]],
                         start[0], start[1]))

        closed: set[Tuple[int, int]] = {start}
        came_from: dict[Tuple[int, int], Tuple[int, int] | None] = {}

        # ----------------------------------------------------------------- #
        # 3️⃣  Main loop
        # ----------------------------------------------------------------- #
        while open_set:
            _, cur_r, cur_c = heapq.heappop(open_set)

            if (cur_r, cur_c) == end:                     # goal reached
                return self._reconstruct_path(came_from,
                                               (cur_r, cur_c))

            for dr, dc in directions:
                nr, nc = cur_r + dr, cur_c + dc
                if self._is_valid(nr, nc) and (nr, nc) not in closed:
                    cost_to_move = self.grid[nr][nc]
                    f_score = self._heuristic((nr, nc), end) + cost_to_move

                    # If we have never seen this node or found a better f‑score
                    if (nr, nc) not in came_from or f_score < came_from[(nr, nc)][1]:
                        came_from[(nr, nc)] = ((cur_r, cur_c), f_score)
                        heapq.heappush(open_set,
                                        (f_score, nr, nc))
                        closed.add((nr, nc))

        # No path found
        return None

    def _is_valid(self, r: int, c: int) -> bool:
        """True if the cell is inside the grid and not a wall."""
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r][c] > 0
        return False

    def _validate_coordinates(self, coord: Tuple[int, int]) -> None:
        """Raise ``ValueError`` for malformed or illegal coordinates."""
        if not (isinstance(coord, tuple) and len(coord) == 2):
            raise ValueError("Coordinates must be a tuple of two integers.")
        r, c = coord
        if self._is_valid(r, c):
            return
        raise ValueError(f"Coordinate {coord} is out of bounds or a wall.")

    def _heuristic(self, node: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Manhattan distance – admissible for this problem."""
        r1, c1 = node
        r2, c2 = goal
        return abs(r1 - r2) + abs(c1 - c2)

    # --------------------------------------------------------------------- #
    # Helper used only by ``find_path`` (private)
    # --------------------------------------------------------------------- #
    @staticmethod
    def _reconstruct_path(
        came_from: dict[Tuple[int, int], Tuple[int, int] | None],
        current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Back‑track from *current* to the start using ``came_from``."""
        path = []
        while current is not None:
            path.append(current)
            current = came_from.get(current)
        return list(reversed(path))


# ------------------------------------------------------------------------- #
# 6 pytest tests
# ------------------------------------------------------------------------- #
import pytest

def test_start_equals_end():
    grid = [[0, 1, 0],
            [1, 1, 1]]
    g = AStarGrid(grid)
    start = (0, 1)
    end = (0, 1)
    assert g.find_path(start, end) == [start]

def test_simple_path():
    grid = [[0, 2, 3],
            [1, 1, 4]]
    g = AStarGrid(grid)
    start = (0, 1)
    end = (1, 2)
    # Down (cost 1) then right (cost 4) → total 7
    path = g.find_path(start, end)
    assert path == [start, (1, 1), end]

def test_wall_blocked():
    grid = [[0, 1],
            [0, 2]]
    g = AStarGrid(grid)
    start = (0, 1)
    end = (1, 1)
    # Direct down is allowed; total cost = 2
    path = g.find_path(start, end)
    assert path == [start, end]

def test_no_path():
    grid = [[0, 0],
            [0, 0]]
    g = AStarGrid(grid)
    start = (0, 1)
    end = (1, 0)
    assert g.find_path(start, end) is None

def test_out_of_bounds_error():
    grid = [[1, 2], [3, 4]]
    g = AStarGrid(grid)
    with pytest.raises(ValueError):
        g.find_path((-1, 0), (0, 0))

def test_rectangular_grid_error_init():
    grid = [[1, 2],
            [3]]          # not rectangular
    with pytest.raises(ValueError):
        AStarGrid(grid)
