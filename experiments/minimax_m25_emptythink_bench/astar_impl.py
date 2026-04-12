"""
A* Pathfinding on a Weighted 2D Grid

This module implements the A* search algorithm on a grid where each cell has a movement cost.
It supports 4-directional movement and uses Manhattan distance as the heuristic.
"""

import heapq
from typing import List, Tuple, Optional


class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.

    Attributes:
        grid (List[List[int]]): 2D list where 0 represents a wall (impassable) and
                                positive integers represent the cost to enter that cell.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the A* grid.

        Args:
            grid: 2D list of movement costs (0 = wall, positive = cost).
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, row: int, col: int) -> bool:
        """Check if a coordinate is within bounds and not a wall."""
        return 0 <= row < self.rows and 0 <= col < self.cols and self.grid[row][col] != 0

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to end using the came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A*.

        Args:
            start: Starting coordinate as (row, col).
            end: Target coordinate as (row, col).

        Returns:
            List of coordinates from start to end inclusive, or None if no path exists.

        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate bounds
        if not self._is_valid(start[0], start[1]) and start != end:
            raise ValueError(f"Start coordinate {start} is out of bounds or is a wall.")
        if not self._is_valid(end[0], end[1]) and start != end:
            raise ValueError(f"End coordinate {end} is out of bounds or is a wall.")

        # Edge case: start equals end
        if start == end:
            return [start]

        # Check if start or end is a wall (if they are not equal to each other)
        if not self._is_valid(start[0], start[1]):
            return None
        if not self._is_valid(end[0], end[1]):
            return None

        # A* initialization
        open_set: List[Tuple[int, Tuple[int, int]]] = []  # (f_score, position)
        heapq.heappush(open_set, (self._manhattan(start, end), start))

        came_from: dict = {}
        g_score: dict = {start: 0}
        f_score: dict = {start: self._manhattan(start, end)}

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)

                if not self._is_valid(neighbor[0], neighbor[1]):
                    continue

                # Cost to enter the neighbor cell
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._manhattan(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        return None
