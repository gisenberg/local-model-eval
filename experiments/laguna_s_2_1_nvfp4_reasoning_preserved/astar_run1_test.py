from typing import List, Tuple, Optional
import heapq
from collections import deque

class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.

    Attributes:
        grid (List[List[int]]): 2D grid where 0 represents walls,
                              positive integers represent weighted cells.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid.

        Args:
            grid: 2D list representing the grid. 0 = wall, positive = weight.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not a wall."""
        row, col = pos
        return (0 <= row < self.rows and
                0 <= col < self.cols and
                self.grid[row][col] != 0)

    def _heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid 4-directional neighbors."""
        row, col = pos
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up
        neighbors = []

        for dr, dc in directions:
            neighbor = (row + dr, col + dc)
            if self._is_valid(neighbor):
                neighbors.append(neighbor)

        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start: Starting position (row, col).
            end: Ending position (row, col).

        Returns:
            List of positions forming the path, or None if no path exists.

        Raises:
            ValueError: If start or end positions are out of bounds or on walls.
        """
        # Validate inputs
        if not self._is_valid(start):
            raise ValueError(f"Invalid start position: {start}")
        if not self._is_valid(end):
            raise ValueError(f"Invalid end position: {end}")

        # Handle trivial case
        if start == end:
            return [start]

        # Initialize data structures
        open_set = [(0, start)]  # (f_score, position)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, end)}

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in self._get_neighbors(current):
                # Calculate tentative g_score
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Found better path to neighbor
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found


# Tests
import pytest

def test_start_equals_end():
    """Test when start equals end."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (0, 0))
    assert result == [(0, 0)]

def test_simple_path():
    """Test finding a simple path."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (2, 2))
    assert result is not None
    assert result[0] == (0, 0)
    assert result[-1] == (2, 2)
    assert len(result) == 5  # Manhattan distance + 1

def test_wall_obstacle():
    """Test path around a wall."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (0, 2))
    assert result is not None
    assert result[0] == (0, 0)
    assert result[-1] == (0, 2)
    # Should go around the wall
    assert len(result) == 7

def test_no_path():
    """Test when no path exists."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (0, 2))
    assert result is None

def test_invalid_start():
    """Test with invalid start position."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((1, 1), (0, 0))  # Start on wall

def test_invalid_end():
    """Test with invalid end position."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))  # End on wall

def test_weighted_optimal_path():
    """Test that algorithm finds optimal weighted path."""
    grid = [
        [1, 1, 1, 1],
        [1, 10, 10, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (0, 3))
    assert result is not None
    # Should prefer lower weight path
    assert result[0] == (0, 0)
    assert result[-1] == (0, 3)