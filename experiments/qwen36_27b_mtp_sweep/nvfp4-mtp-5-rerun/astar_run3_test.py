import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    Uses 4-directional movement, Manhattan heuristic, and heapq for the priority queue.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.
        :param grid: 2D list where 0 represents a wall and positive integers represent traversal costs.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        return self.grid[pos[0]][pos[1]] == 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic (admissible and consistent for 4-directional grids)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.
        :param start: (row, col) tuple
        :param end: (row, col) tuple
        :return: List of (row, col) tuples representing the path, or None if no path exists.
        :raises ValueError: If start or end is out of bounds or is a wall.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds.")
        if self._is_wall(start) or self._is_wall(end):
            raise ValueError("Start or end position is a wall.")

        if start == end:
            return [start]

        # Priority queue: (f_score, tie_breaker_counter, current_position)
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))
        
        g_score = {start: 0}
        came_from = {}
        closed_set = set()
        counter = 1

        while open_set:
            f, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == end:
                # Reconstruct path
                path = []
                curr = end
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.append(start)
                return path[::-1]

            # Explore 4-directional neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)

                if not self._is_valid(neighbor) or self._is_wall(neighbor):
                    continue

                tentative_g = g_score[current] + self.grid[nr][nc]

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1

        return None

import pytest

def test_basic_path():
    """Test standard pathfinding on an open grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Optimal Manhattan distance + 1

def test_start_equals_end():
    """Handle start == end case."""
    assert AStarGrid([[1]]).find_path((0, 0), (0, 0)) == [(0, 0)]

def test_no_path_blocked_by_walls():
    """Return None when walls completely block the path."""
    grid = [[1, 0, 1], [1, 0, 1], [1, 0, 1]]
    assert AStarGrid(grid).find_path((0, 0), (0, 2)) is None

def test_out_of_bounds_raises_value_error():
    """Raise ValueError for coordinates outside grid dimensions."""
    with pytest.raises(ValueError, match="out of bounds"):
        AStarGrid([[1, 1], [1, 1]]).find_path((0, 0), (2, 2))

def test_wall_start_end_raises_value_error():
    """Raise ValueError when start or end is a wall (0)."""
    grid = [[0, 1], [1, 1]]
    with pytest.raises(ValueError, match="wall"):
        AStarGrid(grid).find_path((0, 0), (1, 1))

def test_weighted_optimal_path():
    """Verify A* chooses lower-cost path over shorter geometric path."""
    grid = [
        [1, 10, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    # Direct path cost: 10 + 1 = 11
    # Detour path cost: 1 + 1 + 1 + 1 = 4
    path = AStarGrid(grid).find_path((0, 0), (0, 2))
    expected = [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]
    assert path == expected