from __future__ import annotations

from heapq import heappush, heappop
from typing import List, Tuple, Optional, Dict


class AStarGrid:
    """
    A* path‑finder for a weighted 2‑D grid.

    The grid is a 2‑dimensional list where a value of ``0`` denotes a wall
    (non‑traversable cell) and any positive integer denotes the cost of
    traversing that cell.  Movement is allowed only in the four cardinal
    directions (Manhattan moves).  The heuristic used is the Manhattan
    distance multiplied by the smallest traversable cell cost, which is
    admissible for this movement model.

    Example
    -------
    >>> grid = [[1, 1], [1, 1]]
    >>> astar = AStarGrid(grid)
    >>> astar.find_path((0, 0), (1, 1))
    [(0, 0), (0, 1), (1, 1)]
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Parameters
        ----------
        grid : List[List[int]]
            2‑D list of integers. ``0`` means wall, ``>0`` means traversable
            cell with the given cost.
        """
        self.grid: List[List[int]] = grid
        self.rows: int = len(grid)
        self.cols: int = len(grid[0]) if self.rows else 0

        # Minimum positive cell cost – used to scale the heuristic.
        self._min_cost: int = min(
            (self.grid[i][j] for i in range(self.rows) for j in range(self.cols) if self.grid[i][j] > 0),
            default=1,
        )

    # --------------------------------------------------------------------- #
    #                         PUBLIC API
    # --------------------------------------------------------------------- #
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from *start* to *end*.

        Parameters
        ----------
        start : Tuple[int, int]
            (row, column) coordinate of the starting cell.
        end : Tuple[int, int]
            (row, column) coordinate of the destination cell.

        Returns
        -------
        Optional[List[Tuple[int, int]]]
            List of coordinates from *start* to *end* (inclusive) if a path
            exists, otherwise ``None``.

        Raises
        ------
        ValueError
            If *start* or *end* are out of bounds or are walls.
        """
        # ----- validation ----------------------------------------------------
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds.")
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall.")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall.")

        if start == end:
            return [start]

        # ----- A* initialization ---------------------------------------------
        directions: List[Tuple[int, int]] = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        def heuristic(pos: Tuple[int, int]) -> float:
            """Manhattan distance scaled by the minimum traversable cost."""
            return self._min_cost * (abs(pos[0] - end[0]) + abs(pos[1] - end[1]))

        start_g: float = self.grid[start[0]][start[1]]
        start_f: float = start_g + heuristic(start)

        open_heap: List[Tuple[float, float, int, int]] = []          # (f, g, row, col)
        heappush(open_heap, (start_f, start_g, start[0], start[1]))

        g_score: Dict[Tuple[int, int], float] = {start: start_g}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        counter: int = 0  # tie‑breaker for heap entries

        while open_heap:
            current_f, current_g, cur_r, cur_c = heappop(open_heap)
            current = (cur_r, cur_c)

            if current == end:                     # goal reached
                # reconstruct path
                path: List[Tuple[int, int]] = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for dr, dc in directions:
                nr, nc = cur_r + dr, cur_c + dc

                # out‑of‑bounds check
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                # wall check
                if self.grid[nr][nc] == 0:
                    continue

                tentative_g = current_g + self.grid[nr][nc]

                if tentative_g < g_score.get(current, float('inf')):
                    # This path to neighbor is better – record it.
                    g_score[current] = tentative_g   # keep entry for current (optional)
                    f = tentative_g + heuristic((nr, nc))
                    counter += 1
                    heappush(open_heap, (f, tentative_g, nr, nc))
                    came_from[(nr, nc)] = current

        # No path found
        return None

import pytest


def test_start_end_same():
    """Start and end are the same cell – should return a list containing that cell."""
    grid = [[1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]


def test_start_is_wall():
    """Starting cell is a wall – should raise ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))


def test_end_is_wall():
    """Ending cell is a wall – should raise ValueError."""
    grid = [[1, 1], [1, 0]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))


def test_start_out_of_bounds():
    """Start position outside the grid – should raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((2, 0), (0, 0))


def test_simple_grid_path():
    """Simple 3×3 grid with a single wall – path must avoid the wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # No wall cells in the returned path
    assert all(grid[r][c] != 0 for r, c in path)
    # Manhattan distance is 4 → path length (including start) is 5
    assert len(path) == 5


def test_weighted_grid_optimal_path():
    """Weighted cells – the algorithm must pick the cheaper route."""
    grid = [
        [1, 100, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    # The optimal path avoids the costly (0,1) cell
    assert (0, 1) not in path
    # Total cost should be 5 (1 + 1 + 1 + 1 + 1)
    total_cost = sum(grid[r][c] for r, c in path)
    assert total_cost == 5
    # Path length is 5 cells
    assert len(path) == 5