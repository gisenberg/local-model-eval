from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    """
    A class to represent a weighted 2D grid for A* pathfinding.

    Attributes:
        grid: A 2D list representing the grid, where each cell's value is the movement cost.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.

    Methods:
        find_path: Finds the shortest path from start to end using A*.
    """

    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from start to end using A* algorithm.

        Args:
            start: Starting position as a tuple (row, col).
            end: Target position as a tuple (row, col).

        Returns:
            List of positions (tuples) representing the path from start to end.
            Returns None if no path exists or if start/end is out of bounds.

        Raises:
            ValueError: If start or end is out of bounds.
        """
        # Validate start and end positions
        if not self._is_valid_position(start) or not self._is_valid_position(end):
            raise ValueError("Start or end is out of bounds.")

        if start == end:
            return [start]

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Priority queue: (f_score, g_score, row, col, path)
        open_set = []
        heapq.heappush(open_set, (0, 0, start[0], start[1], [start]))

        # Visited set to avoid revisiting
        visited = set()
        visited.add((start[0], start[1]))

        # To store the cost to reach each cell
        g_score = {start: 0}

        while open_set:
            _, current_g, row, col, path = heapq.heappop(open_set)

            # If we reached the end
            if (row, col) == end:
                return path

            # Explore neighbors
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc

                # Check if the new position is valid
                if not self._is_valid_position((new_row, new_col)) or (new_row, new_col) in visited:
                    continue

                # Skip impassable cells (cost 0)
                if self.grid[new_row][new_col] == 0:
                    continue

                # Calculate new cost
                new_g = current_g + self.grid[new_row][new_col]
                new_f = new_g + self._manhattan_heuristic((new_row, new_col), end)

                # If this path to the new cell is better, update
                if (new_row, new_col) not in g_score or new_g < g_score[(new_row, new_col)]:
                    g_score[(new_row, new_col)] = new_g
                    heapq.heappush(open_set, (new_f, new_g, new_row, new_col, path + [(new_row, new_col)]))
                    visited.add((new_row, new_col))

        # No path found
        return None

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within the grid bounds."""
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _manhattan_heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

import pytest


def test_simple_path():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    grid = [[1 if i != 1 or j != 1 else 0 for j in range(3)] for i in range(3)]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

def test_no_path():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    grid = [[1 if i != 1 or j != 1 else 0 for j in range(3)] for i in range(3)]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 1))
    assert path is None

def test_start_end_same():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_out_of_bounds():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((3, 0), (0, 0))

def test_walls():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

def test_optimal_cost():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]