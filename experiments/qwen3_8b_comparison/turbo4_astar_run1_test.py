from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    """
    A class to represent a weighted 2D grid for A* pathfinding.
    Each cell has a movement cost (0 = impassable, positive int = cost to enter).
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid.

        Args:
            grid: A 2D list of integers representing the grid.
                  0 = impassable (wall), positive int = movement cost.
        """
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]:
        """
        Find the shortest path from start to end using A* algorithm.

        Args:
            start: Starting position (row, col)
            end: Target position (row, col)

        Returns:
            List of positions (tuples) representing the path, or None if no path exists.
        """
        # Validate inputs
        if not self._is_valid_position(start) or not self._is_valid_position(end):
            raise ValueError("Start or end position is out of bounds.")
        if start == end:
            return [start]

        # Initialize open and closed sets
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, end)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct the path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                tentative_g_score = g_score[current] + self._cost(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        return None

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within the grid bounds."""
        row, col = pos
        return 0 <= row < self.height and 0 <= col < self.width

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all valid 4-directional neighbors of a position."""
        row, col = pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_position((new_row, new_col)) and self.grid[new_row][new_col] != 0:
                neighbors.append((new_row, new_col))
        return neighbors

    def _cost(self, current: Tuple[int, int], neighbor: Tuple[int, int]) -> int:
        """Return the movement cost from current to neighbor."""
        return self.grid[neighbor[0]][neighbor[1]]

    def _heuristic(self, current: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Use Manhattan distance as heuristic."""
        return abs(current[0] - end[0]) + abs(current[1] - end[1])

import pytest
from typing import List, Tuple

def test_grid():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    return AStarGrid(grid)

def test_simple_path():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = test_grid()
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

def test_blocked_path():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = test_grid()
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_start_end_same():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = test_grid()
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_out_of_bounds():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = test_grid()
    with pytest.raises(ValueError):
        astar.find_path((3, 0), (0, 0))

def test_invalid_start_end():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = test_grid()
    with pytest.raises(ValueError):
        astar.find_path((0, 3), (0, 0))

def test_custom_grid():
    grid = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    astar = test_grid()
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]