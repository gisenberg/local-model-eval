import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Supports 4-directional movement, Manhattan heuristic, and weighted cells.
    Walls are represented by 0, and passable cells have positive weights/costs.
    The cost of a path is the sum of the weights of the cells entered (excluding start).
    """

    def __init__(self, grid: List[List[float]]):
        """
        Initialize the grid.
        :param grid: 2D list where 0 represents a wall, and positive numbers represent cell weights.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.
        
        :param start: (row, col) tuple
        :param end: (row, col) tuple
        :return: List of (row, col) tuples representing the optimal path, or None if unreachable.
        :raises ValueError: If start or end is out of bounds or on a wall.
        """
        if start == end:
            return [start]

        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds or on a wall.")

        open_set = []
        counter = 0
        initial_h = self._heuristic(start, end)
        heapq.heappush(open_set, (initial_h, counter, start))
        counter += 1

        g_score = {start: 0.0}
        came_from = {}

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_set:
            f, _, current = heapq.heappop(open_set)

            # Skip stale entries in the priority queue
            if f > g_score.get(current, float('inf')):
                continue

            if current == end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._is_valid(neighbor):
                    continue

                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1
                    came_from[neighbor] = current

        return None

import pytest

def test_basic_path():
    """Test simple pathfinding on an unweighted grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Manhattan distance (4 steps) + 1 start node

def test_weighted_optimal():
    """Test that A* correctly avoids high-weight cells when a cheaper path exists."""
    grid = [
        [1, 100, 100],
        [1, 100, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Verify the path avoids all high-cost cells
    assert all(grid[r][c] != 100 for r, c in path)

def test_no_path():
    """Test that None is returned when the destination is completely blocked."""
    grid = [[1, 0, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None

def test_start_equals_end():
    """Test edge case where start and end are the same position."""
    grid = [[1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_out_of_bounds():
    """Test that ValueError is raised for coordinates outside the grid."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_wall_at_start_or_end():
    """Test that ValueError is raised when start or end lies on a wall."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))