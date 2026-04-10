from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    """
    A class to represent a weighted 2D grid for A* pathfinding.
    
    Attributes:
        grid: A 2D list representing the grid where each cell has a movement cost.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.

        Args:
            grid: A 2D list where each cell represents the movement cost.
                  0 means impassable (wall), positive integers are movement cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.

        Args:
            start: Starting position as a tuple (row, col).
            end: Ending position as a tuple (row, col).

        Returns:
            List of coordinates representing the shortest path from start to end.
            Returns None if no path exists or if start/end is invalid.
        """
        # Validate start and end positions
        if not self._is_valid_position(start) or not self._is_valid_position(end):
            raise ValueError("Start or end position is out of bounds or invalid.")
        if start == end:
            return [start]

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Priority queue: (f_score, g_score, current_position, path)
        open_set = []
        heapq.heappush(open_set, (0, 0, start, [start]))

        # Visited set to avoid revisiting nodes
        visited = set()
        # To track the cost to reach each node
        g_score = {start: 0}
        # To track the best estimated total cost (f = g + h)
        f_score = {start: self._manhattan_heuristic(start, end)}

        while open_set:
            _, current_g, current_pos, path = heapq.heappop(open_set)

            if current_pos == end:
                return path

            if current_pos in visited:
                continue

            visited.add(current_pos)

            for dr, dc in directions:
                r, c = current_pos
                nr, nc = r + dr, c + dc

                if not self._is_valid_position((nr, nc)):
                    continue

                if self.grid[nr][nc] == 0:
                    continue  # Wall, impassable

                new_g = current_g + self.grid[nr][nc]
                new_pos = (nr, nc)

                if new_pos not in g_score or new_g < g_score[new_pos]:
                    g_score[new_pos] = new_g
                    h = self._manhattan_heuristic(new_pos, end)
                    f = new_g + h
                    f_score[new_pos] = f
                    heapq.heappush(open_set, (f, new_g, new_pos, path + [new_pos]))

        # No path found
        return None

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Check if a position is within the grid bounds and not a wall.
        """
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != 0

    def _manhattan_heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """
        Calculate the Manhattan distance heuristic between two positions.
        """
        r1, c1 = pos
        r2, c2 = end
        return abs(r1 - r2) + abs(c1 - c2)

import pytest


def test_valid_path():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    grid_obj = AStarGrid(grid)
    path = grid_obj.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]

def test_no_path():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    grid_obj = AStarGrid(grid)
    path = grid_obj.find_path((0, 0), (2, 2))
    assert path is not None  # Should have a path
    grid[1][1] = 0  # Make it a wall
    path = grid_obj.find_path((0, 0), (2, 2))
    assert path is None

def test_start_end_same():
    grid = [[1, 1], [1, 1]]
    grid_obj = AStarGrid(grid)
    path = grid_obj.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_out_of_bounds():
    grid = [[1, 1], [1, 1]]
    grid_obj = AStarGrid(grid)
    with pytest.raises(ValueError):
        grid_obj.find_path((3, 0), (0, 0))

def test_wall():
    grid = [[1, 1], [1, 0]]
    grid_obj = AStarGrid(grid)
    with pytest.raises(ValueError):
        grid_obj.find_path((0, 0), (1, 1))

def test_optimal_path():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    grid_obj = AStarGrid(grid)
    path = grid_obj.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]