import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.

    Attributes:
        grid: 2D list where 0 represents a wall and positive numbers represent weights.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D grid.

        Args:
            grid: 2D list where 0 represents walls and positive numbers represent weights.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance heuristic.

        Args:
            pos: Current position (row, col).
            end: Target position (row, col).

        Returns:
            Manhattan distance between pos and end.
        """
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    def _is_valid(self, row: int, col: int) -> bool:
        """
        Check if a position is within bounds and not a wall.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            True if position is valid, False otherwise.
        """
        return (0 <= row < self.rows and
                0 <= col < self.cols and
                self.grid[row][col] != 0)

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start: Starting position (row, col).
            end: Target position (row, col).

        Returns:
            List of positions forming the path, or None if no path exists.

        Raises:
            ValueError: If start or end positions are out of bounds.
        """
        # Validate inputs
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start position {start} is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End position {end} is out of bounds")

        # Handle same start and end
        if start == end:
            return [start]

        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None

        # Initialize data structures
        open_heap = []
        heapq.heappush(open_heap, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, end)}

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_heap:
            current_f, current = heapq.heappop(open_heap)

            # Found the goal
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            # Skip if we've found a better path to this node
            if current_f > f_score.get(current, float('inf')):
                continue

            # Explore neighbors
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)

                if not self._is_valid(neighbor[0], neighbor[1]):
                    continue

                # Calculate tentative g_score
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                # If this path is better, update
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))

        # No path found
        return None


# Pytest tests
import pytest

def test_same_start_end():
    """Test when start equals end."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (0, 0))
    assert result == [(0, 0)]

def test_simple_path():
    """Test a simple 2x2 grid with no obstacles."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (1, 1))
    assert result == [(0, 0), (1, 1)] or result == [(0, 0), (0, 1), (1, 1)]

def test_wall_obstacle():
    """Test pathfinding around a wall."""
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

def test_no_path():
    """Test when no path exists due to walls."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (2, 2))
    assert result is None

def test_out_of_bounds():
    """Test that out of bounds raises ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))

def test_weighted_path():
    """Test that A* finds optimal path considering weights."""
    # Grid where going around is cheaper than through high-weight cells
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (0, 2))
    assert result is not None
    # Should go down, right, right, up instead of right, right
    assert result == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]