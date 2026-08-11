import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """A* pathfinding on a weighted 2D grid with 4-directional movement."""

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.

        Args:
            grid: 2D list where 0 represents a wall (impassable) and
                  positive integers represent movement cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0

    def _heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    def _is_valid(self, row: int, col: int) -> bool:
        """Check if position is within bounds and not a wall."""
        return (0 <= row < self.rows and
                0 <= col < self.cols and
                self.grid[row][col] != 0)

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start: Starting position as (row, col)
            end: Target position as (row, col)

        Returns:
            List of positions forming the path, or None if no path exists

        Raises:
            ValueError: If start or end positions are out of bounds or are walls
        """
        # Validate inputs
        if not self._is_valid(*start) or self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is invalid")
        if not self._is_valid(*end) or self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is invalid")

        # Handle same start and end
        if start == end:
            return [start]

        # Initialize data structures
        open_heap = [(0, start)]  # (f_score, position)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, end)}

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_heap:
            current_f, current = heapq.heappop(open_heap)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Skip if we've found a better path to this node
            if current_f > f_score.get(current, float('inf')):
                continue

            # Explore neighbors
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)

                if not self._is_valid(*neighbor):
                    continue

                # Calculate tentative g_score
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                # If this path is better, update
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor))

        return None  # No path found


# Tests
import pytest

def test_simple_path():
    """Test basic pathfinding on simple grid."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Manhattan distance + 1

def test_same_start_end():
    """Test when start equals end."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_wall_blocking():
    """Test pathfinding around walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (0, 2)

def test_no_path():
    """Test when no path exists due to walls."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_weighted_path():
    """Test that algorithm chooses lower-cost path."""
    # Two paths: one cheap, one expensive
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    # Should go around the expensive middle column
    assert (1, 0) in path or (2, 0) in path
    assert (1, 2) in path or (2, 2) in path

def test_out_of_bounds():
    """Test that out-of-bounds raises ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))