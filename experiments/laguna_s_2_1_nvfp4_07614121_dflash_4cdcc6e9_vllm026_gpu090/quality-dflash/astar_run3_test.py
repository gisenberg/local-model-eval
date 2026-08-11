import heapq
from typing import List, Optional, Tuple

class AStarGrid:
    """A* pathfinding on a weighted 2D grid with 4-directional movement."""

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.

        Args:
            grid: 2D list where 0 represents walls and positive integers represent weights.
                  Negative values are treated as walls.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, x: int, y: int) -> bool:
        """Check if coordinates are within bounds and not a wall."""
        return (0 <= x < self.rows and
                0 <= y < self.cols and
                self.grid[x][y] > 0)

    def _heuristic(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """Manhattan distance heuristic."""
        return abs(x1 - x2) + abs(y1 - y2)

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from start to end."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.

        Args:
            start: Starting coordinates (row, col)
            end: Ending coordinates (row, col)

        Returns:
            List of coordinates representing the path, or None if no path exists.

        Raises:
            ValueError: If start or end is out of bounds or on a wall.
        """
        # Validate inputs
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position out of bounds")
        if self.grid[start[0]][start[1]] <= 0:
            raise ValueError("Start position is a wall")
        if self.grid[end[0]][end[1]] <= 0:
            raise ValueError("End position is a wall")

        # Handle same start and end
        if start == end:
            return [start]

        # Initialize data structures
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start[0], start[1], end[0], end[1])}

        # 4-directional movement: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            # Skip if we've already found a better path
            if current_f > f_score.get(current, float('inf')):
                continue

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                if not self._is_valid(neighbor[0], neighbor[1]):
                    continue

                # Calculate tentative g_score
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(
                        neighbor[0], neighbor[1], end[0], end[1]
                    )
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None  # No path found


# Tests
import pytest

def test_same_start_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (0, 0))
    assert result == [(0, 0)]

def test_simple_path():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (2, 2))
    expected = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert result == expected

def test_wall_blocking():
    grid = [[1, 0, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (0, 2))
    assert result is None

def test_weighted_cells():
    grid = [[1, 1, 1], [1, 10, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (2, 2))
    # Should go around the expensive middle cell
    assert result == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]

def test_out_of_bounds():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_wall_positions():
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((1, 1), (0, 0))