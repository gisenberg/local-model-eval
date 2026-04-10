from __future__ import annotations
import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the grid. 0 = impassable wall, positive int = cost to enter."""
        self.grid = grid
        self.rows = len(grid)
        if self.rows == 0:
            raise ValueError("grid must have at least one row")
        self.cols = len(grid[0])
        for r in range(self.rows):
            if len(grid[r]) != self.cols:
                raise ValueError("rectangular grid required")

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Return the optimal path from start to end using A* with Manhattan heuristic.
        
        Args:
            start: (row, col) coordinates of the source cell.
            end:   (row, col) coordinates of the destination cell.

        Returns:
            List of (row, col) tuples representing the path, or None if no path exists.
            
        Raises:
            ValueError: If start or end is out of bounds, not a valid cell,
                        or both are walls. Also if grid is empty.
        """
        # Validate inputs
        self._validate_coordinates(start)
        self._validate_coordinates(end)

        if start == end:
            return [start]

        rows, cols = self.rows, self.cols

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Open set as min-heap of (f_score, row, col)
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, end) + self.grid[start[0]][start[1]], start[0], start[1]))

        # Closed set for fast lookup
        closed: set[Tuple[int, int]] = {start}

        # Predecessor map to reconstruct path
        came_from: dict[Tuple[int, int], Tuple[int, int] | None] = {}

        while open_set:
            _, cur_r, cur_c = heapq.heappop(open_set)

            if (cur_r, cur_c) == end:
                # Reconstruct path
                path = []
                while (cur_r, cur_c) is not None:
                    path.append((cur_r, cur_c))
                    cur_r, cur_c = came_from.get((cur_r, cur_c), None)
                return list(reversed(path))

            for dr, dc in directions:
                nr, nc = cur_r + dr, cur_c + dc
                if self._is_valid(nr, nc) and (nr, nc) not in closed:
                    # Cost to move into this cell is its grid value
                    cost_to_move = self.grid[nr][nc]
                    f_score = self._heuristic((nr, nc), end) + cost_to_move

                    if (nr, nc) not in came_from or f_score < came_from[(nr, nc)][1]:
                        came_from[(nr, nc)] = ((cur_r, cur_c), f_score)
                        heapq.heappush(open_set, (f_score, nr, nc))
                        closed.add((nr, nc))

        # No path found
        return None

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within grid bounds and not a wall."""
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r][c] > 0
        return False

    def _validate_coordinates(self, coord: Tuple[int, int]) -> None:
        """Raise ValueError for out-of-bounds or wall coordinates."""
        if not (isinstance(coord, tuple) and len(coord) == 2):
            raise ValueError("Coordinates must be a tuple of two integers.")
        r, c = coord
        if self._is_valid(r, c):
            return
        raise ValueError(f"Coordinate {coord} is out of bounds or a wall.")

    def _heuristic(self, node: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Manhattan distance heuristic (ignores weights)."""
        r1, c1 = node
        r2, c2 = goal
        return abs(r1 - r2) + abs(c1 - c2)
