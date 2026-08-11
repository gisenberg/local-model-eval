import heapq
from typing import List, Optional, Tuple, Set

class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.

    Attributes:
        grid (List[List[int]]): The 2D grid where 0 represents a wall,
                                and any positive number represents the cost of moving through that cell.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the AStarGrid with a given grid.

        Args:
            grid: A 2D list of integers representing the grid.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, row: int, col: int) -> bool:
        """Checks if a cell is within the grid boundaries and not a wall."""
        return 0 <= row < self.rows and 0 <= col < self.cols and self.grid[row][col] != 0

    def _heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Calculates the Manhattan distance heuristic between two points."""
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from start to end using the A* algorithm.

        Args:
            start: The starting position as a tuple (row, col).
            end: The ending position as a tuple (row, col).

        Returns:
            A list of tuples representing the path from start to end, or None if no path exists.

        Raises:
            ValueError: If the start or end positions are out of bounds.
        """
        if not self._is_valid(start[0], start[1]):
            raise ValueError(f"Start position {start} is out of bounds or a wall.")
        if not self._is_valid(end[0], end[1]):
            raise ValueError(f"End position {end} is out of bounds or a wall.")

        if start == end:
            return [start]

        # Priority queue: (f_score, g_score, position)
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))

        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}

        g_score: dict[Tuple[int, int], int] = {start: 0}
        f_score: dict[Tuple[int, int], int] = {start: self._heuristic(start, end)}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Reverse to get path from start to end

            # Explore neighbors
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)

                if not self._is_valid(neighbor[0], neighbor[1]):
                    continue

                tentative_g_score = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))

        return None  # No path found

import pytest

# Test grid for testing
GRID = [
    [1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 0, 1],
    [1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1],
]

# Weighted grid for testing
WEIGHTED_GRID = [
    [1, 1, 1, 1, 1],
    [1, 9, 9, 9, 1],
    [1, 9, 1, 9, 1],
    [1, 9, 9, 9, 1],
    [1, 1, 1, 1, 1],
]

def test_simple_path():
    """Test finding a simple path in a grid with walls."""
    grid = AStarGrid(GRID)
    path = grid.find_path((0, 0), (4, 4))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
    # Check that the path is valid (each step is adjacent and not a wall)
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1
        assert GRID[r1][c1] != 0 and GRID[r2][c2] != 0

def test_start_equals_end():
    """Test that the path is just the start point when start equals end."""
    grid = AStarGrid(GRID)
    path = grid.find_path((2, 2), (2, 2))
    assert path == [(2, 2)]

def test_no_path():
    """Test that None is returned when there is no path due to walls."""
    grid = AStarGrid(GRID)
    # Path from top-left to bottom-right of the enclosed area is blocked
    path = grid.find_path((1, 1), (3, 3))
    assert path is None

def test_optimal_path_with_weights():
    """Test that the algorithm finds the optimal path in a weighted grid."""
    grid = AStarGrid(WEIGHTED_GRID)
    path = grid.find_path((0, 0), (4, 4))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)

    # Calculate total cost of the found path
    total_cost = sum(WEIGHTED_GRID[r][c] for r, c in path)

    # Manually find the cheapest path for comparison
    # In this grid, the cheapest path avoids the 9-cost cells
    expected_path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (4, 4)]
    expected_cost = sum(WEIGHTED_GRID[r][c] for r, c in expected_path)
    assert total_cost == expected_cost

def test_out_of_bounds_start():
    """Test that a ValueError is raised for an out-of-bounds start."""
    grid = AStarGrid(GRID)
    with pytest.raises(ValueError):
        grid.find_path((-1, 0), (4, 4))

def test_out_of_bounds_end():
    """Test that a ValueError is raised for an out-of-bounds end."""
    grid = AStarGrid(GRID)
    with pytest.raises(ValueError):
        grid.find_path((0, 0), (5, 5))