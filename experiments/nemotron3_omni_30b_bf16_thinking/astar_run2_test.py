"""A* path‑finding on a weighted 2‑D grid.

The grid is a list of lists where a value of ``0`` means a wall (impassable)
and any positive number is a walkable cell whose value is the cost of entering
that cell.  The algorithm uses the 4‑directional Manhattan distance as the
heuristic and a binary heap (`heapq`) for the open set, guaranteeing an
optimal (least‑cost) path when all costs are non‑negative.
"""

from __future__ import annotations

import heapq
from typing import List, Tuple, Optional, Dict


class AStarGrid:
    """
    A* path finder for a weighted 2‑D grid.

    Parameters
    ----------
    grid : List[List[float]]
        2‑D list representing the map. ``0`` denotes a wall, any positive
        number is a walkable cell with the given cost.

    Notes
    -----
    * Only the four orthogonal neighbours (up, down, left, right) are
      considered.
    * The heuristic used is the Manhattan distance, which is admissible for
      4‑directional movement with non‑negative cell costs, thus guaranteeing
      optimality.
    """

    def __init__(self, grid: List[List[float]]) -> None:
        self.grid: List[List[float]] = grid
        self rows: int = len(grid)
        self cols: int = len(grid[0]) if self rows else 0

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from ``start`` to ``end`` on the grid.

        Parameters
        ----------
        start : Tuple[int, int]
            (row, column) coordinates of the starting cell.
        end : Tuple[int, int]
            (row, column) coordinates of the destination cell.

        Returns
        -------
        Optional[List[Tuple[int, int]]]
            A list of coordinates from ``start`` to ``end`` inclusive, or
            ``None`` if no path exists (this only happens when the start and
            end are in different connected components).

        Raises
        ------
        ValueError
            * If ``start`` or ``end`` are out of bounds.
            * If ``start`` or ``end`` lie on a wall (grid value ``0``).
        """
        # ---- basic validation ------------------------------------------------
        for point, name in ((start, "start"), (end, "end")):
            if not self._in_bounds(point):
                raise ValueError(f"{name} coordinates {point} are out of bounds")
            row, col = point
            if self.grid[row][col] == 0:
                raise ValueError(f"{name} coordinates {point} are on a wall")

        if start == end:
            return [start]

        # ---- A* initialisation ------------------------------------------------
        open_heap: List[Tuple[float, int, Tuple[int, int]]] = []   # (f, count, node)
        heapq.heappush(open_heap, (self._heuristic(start, end), 0, start))

        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # ``counter`` guarantees a deterministic order when f_scores are equal
        counter = 0

        # ---- main loop -------------------------------------------------------
        while open_heap:
            current_f, _, current = heapq.heappop(open_heap)

            if current == end:
                return self._reconstruct_path(came_from, start, end)

            for neighbor in self._neighbors(current):
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_heap, (f, counter, neighbor))

        # No path found
        return None

    # --------------------------------------------------------------------- #
    # Private helpers
    # --------------------------------------------------------------------- #
    def _in_bounds(self, pt: Tuple[int, int]) -> bool:
        """Return ``True`` if ``pt`` lies inside the grid."""
        rows, cols = self.rows, self.cols
        r, c = pt
        return 0 <= r < rows and 0 <= c < cols

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance between two grid points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _neighbors(self, pt: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return the 4‑directional neighbours of ``pt`` that are walkable."""
        rows, cols = self.rows, self.cols
        r, c = pt
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        result = []
        for nr, nc in candidates:
            if self._in_bounds((nr, nc)) and self.grid[nr][nc] != 0:
                result.append((nr, nc))
        return result

    @staticmethod
    def _reconstruct_path(came_from: Dict[Tuple[int, int], Tuple[int, int]],
                          start: Tuple[int, int],
                          goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from ``start`` to ``goal`` using ``came_from``."""
        path: List[Tuple[int, int]] = [goal]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return path

import pytest
from typing import List, Tuple, Optional


def test_start_equals_end():
    grid = [[1, 2], [3, 4]]
    finder = AStarGrid(grid)
    assert finder.find_path((0, 0), (0, 0)) == [(0, 0)]


def test_wall_start():
    grid = [[0, 1], [1, 1]]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError, match="on a wall"):
        finder.find_path((0, 0), (1, 1))


def test_end_out_of_bounds():
    grid = [[1, 1], [1, 1]]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        finder.find_path((0, 0), (2, 2))   # row 2 does not exist


def test_simple_grid():
    """Start and end are neighbours – the shortest path has two cells."""
    grid = [[1, 1], [1, 1]]
    finder = AStarGrid(grid)
    path = finder.find_path((0, 0), (1, 1))
    assert isinstance(path, list)
    assert len(path) == 3                     # start + 2 steps
    assert path[0] == (0, 0)
    assert path[-1] == (1, 1)

    # verify optimality: total cost = 1 + 1 = 2
    total_cost = sum(grid[r][c] for r, c in path[1:])   # exclude start cell
    assert total_cost == 2


def test_wall_forces_route():
    """
    Grid layout (0 = wall):
        1 0 1
        1 1 1
        0 1 1
    The only viable route is:
        (0,0) → (1,0) → (1,1) → (1,2) → (2,2)
    Total cost = 1+1+1+1 = 4.
    """
    grid = [
        [1, 0, 1],
        [1, 1, 1],
        [0, 1, 1],
    ]
    finder = AStarGrid(grid)
    path = finder.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5                     # 4 moves
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

    # verify cost
    total = sum(grid[r][c] for r, c in path[1:])
    assert total == 4


def test_optimal_weighted_path():
    """
    Two possible routes from (0,0) to (2,2):
        1) (0,0)->(0,1)->(0,2)->(1,2)->(2,2)  cost = 2+1+1+1 = 5
        2) (0,0)->(1,0)->(2,0)->(2,1)->(2,2)  cost = 1+1+1+1 = 4
    The algorithm must pick the cheaper one.
    """
    grid = [
        [1, 2, 1],
        [1, 0, 1],
        [1, 1, 1],
    ]
    finder = AStarGrid(grid)
    path = finder.find_path((0, 0), (2, 2))
    assert path is not None

    # verify that the chosen path is the cheaper one (cost = 4)
    total = sum(grid[r][c] for r, c in path[1:])
    assert total == 4

    # also ensure the path respects wall cells (value 0)
    for r, c in path:
        assert grid[r][c] != 0