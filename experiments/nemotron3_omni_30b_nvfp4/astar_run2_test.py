from __future__ import annotations

import heapq
from typing import List, Tuple, Optional


class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.

    Class Attributes:
        grid (List[List[int]]): 2D grid where 0 represents a wall and positive integers represent weighted cells.
        start (Tuple[int, int]): Starting position (row, column).
        end (Tuple[int, int]): Ending position (row, column).

    Methods:
        find_path(start, end) -> Optional[List[Tuple[int, int]]]:
            Finds the optimal path from start to end using A* algorithm.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a given 2D grid.

        Args:
            grid (List[List[int]]): 2D grid where 0 represents a wall and positive integers represent weighted cells.

        Raises:
            ValueError: If the grid is empty or not rectangular.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")

        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

        for row in grid:
            if len(row) != self.cols:
                raise ValueError("Grid must be rectangular.")

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start (Tuple[int, int]): Starting position (row, column).
            end (Tuple[int, int]): Ending position (row, column).

        Returns:
            Optional[List[Tuple[int, int]]]: List of positions representing the path from start to end, or None if no path exists.

        Raises:
            ValueError: If start or end positions are out of bounds.
        """
        if start == end:
            return [start]

        if not self._is_valid_position(start) or not self._is_valid_position(end):
            raise ValueError("Start or end position is out of bounds.")

        open_set = []
        heapq.heappush(open_set, self._heuristic(start, end), start)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, end)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._get_neighbors(current):
                tentative_g_score = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, f_score[neighbor], neighbor)

        return None

    def _heuristic(self, start: Tuple[int, int], end: Tuple[int, int]) -> int:
        """
        Calculate the Manhattan distance between start and end positions.

        Args:
            start (Tuple[int, int]): Starting position (row, column).
            end (Tuple[int, int]): Ending position (row, column).

        Returns:
            int: Manhattan distance between start and end.
        """
        return abs(start[0] - end[0]) + abs(start[1] - end[1])

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Check if a position is within the grid bounds and not a wall.

        Args:
            pos (Tuple[int, int]): Position (row, column) to check.

        Returns:
            bool: True if the position is valid, False otherwise.
        """
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols and self.grid[row][col] != 0

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get the 4-directional neighbors of a position.

        Args:
            pos (Tuple[int, int]): Position (row, column) to get neighbors for.

        Returns:
            List[Tuple[int, int]]: List of neighboring positions.
        """
        row, col = pos
        neighbors = [(row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)]
        return [pos for pos in neighbors if self._is_valid_position(pos)]

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from start to current position.

        Args:
            came_from (dict): Mapping of positions to their previous positions.
            current (Tuple[int, int]): Current position.

        Returns:
            List[Tuple[int, int]]: List of positions representing the path from start to current.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


# Pytest tests
def test_a_star_grid_find_path_start_end_same():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    a_star_grid = AStarGrid(grid)
    path = a_star_grid.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_a_star_grid_find_path_wall():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    a_star_grid = AStarGrid(grid)
    with pytest.raises(ValueError):
        a_star_grid.find_path((0, 0), (1, 1))

def test_a_star_grid_find_path_out_of_bounds():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    a_star_grid = AStarGrid(grid)
    with pytest.raises(ValueError):
        a_star_grid.find_path((0, 0), (3, 3))

def test_a_star_grid_find_path_no_path():
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    a_star_grid = AStarGrid(grid)
    path = a_star_grid.find_path((0, 0), (2, 2))
    assert path is None

def test_a_star_grid_find_path_weighted_grid():
    grid = [
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ]
    a_star_grid = AStarGrid(grid)
    path = a_star_grid.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

def test_a_star_grid_find_path_weighted_grid_alternative_path():
    grid = [
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ]
    a_star_grid = AStarGrid(grid)
    path = a_star_grid.find_path((0, 0), (2, 2))
    assert path in [
        [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)],
        [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    ]