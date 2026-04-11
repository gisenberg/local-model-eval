from __future__ import annotations

import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* path‑finding on a 2‑D grid with movement costs.

    The grid is a list of lists where:
        * 0   → impassable wall
        * >0  → cost to *enter* that cell
    Movement is allowed in the four cardinal directions.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Store the grid and pre‑compute its dimensions.

        Args:
            grid: 2‑D list of non‑negative integers representing movement cost.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non‑empty")
        self._grid: List[List[int]] = grid
        self._rows: int = len(grid)
        self._cols: int = len(grid[0])

    # ------------------------------------------------------------------ #
    # Helper methods
    # ------------------------------------------------------------------ #
    def _in_bounds(self, cell: Tuple[int, int]) -> bool:
        r, c = cell
        return 0 <= r < self._rows and 0 <= c < self._cols

    def _is_wall(self, cell: Tuple[int, int]) -> bool:
        r, c = cell
        return self._grid[r][c] == 0

    def _neighbors(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = cell
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        return [nb for nb in candidates if self._in_bounds(nb) and not self._is_wall(nb)]

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def find_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Return the cheapest path from *start* to *end* (inclusive) using A*,
        or ``None`` if no path exists.

        The total cost of a path is the sum of the movement costs of all
        entered cells (the start cell is **not** charged because you begin
        there).

        Raises:
            ValueError: if *start* or *end* lies outside the grid.
        """
        # ----- validation -------------------------------------------------
        if not self._in_bounds(start) or not self._in_bounds(end):
            raise ValueError("Start or end coordinate is out of bounds")
        if self._is_wall(start) or self._is_wall(end):
            # If start == end and it is a wall we also return None
            return None
        if start == end:
            return [start]

        # ----- A* core ----------------------------------------------------
        open_heap: List[Tuple[int, int, int, Tuple[int, int]]] = []  # (f, g, counter, node)
        counter = 0  # tie‑breaker for heapq
        heapq.heappush(open_heap, (0, 0, counter, start))
        counter += 1

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        f_score: Dict[Tuple[int, int], int] = {
            start: self._manhattan(start, end)
        }

        while open_heap:
            _, g_current, _, current = heapq.heappop(open_heap)

            if current == end:
                # reconstruct path
                path: List[Tuple[int, int]] = [end]
                while path[-1] != start:
                    path.append(came_from[path[-1]])
                path.reverse()
                return path

            for neighbor in self._neighbors(current):
                tentative_g = g_score[current] + self._grid[neighbor[0]][neighbor[1]]
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._manhattan(neighbor, end)
                    f_score[neighbor] = f
                    heapq.heappush(open_heap, (f, tentative_g, counter, neighbor))
                    counter += 1

        # open set exhausted → no path
        return None
