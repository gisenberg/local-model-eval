import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    Supports 4-directional movement, Manhattan heuristic, and cell weights.
    Walls are represented by 0, and positive numbers represent traversal costs.
    """

    def __init__(self, grid: List[List[float]]):
        """
        Initialize the grid.
        :param grid: 2D list where 0 represents a wall and positive numbers represent traversal costs.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_in_bounds(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return self.grid[r][c] == 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic (admissible and consistent for 4-directional movement)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        :param start: (row, col) tuple for the starting position.
        :param end: (row, col) tuple for the target position.
        :return: List of (row, col) tuples representing the optimal path, or None if unreachable.
        :raises ValueError: If start or end is out of bounds or on a wall.
        """
        # Validate start and end positions
        if not self._is_in_bounds(start) or not self._is_in_bounds(end):
            raise ValueError("Start or end position is out of bounds.")
        if self._is_wall(start) or self._is_wall(end):
            raise ValueError("Start or end position is on a wall.")

        if start == end:
            return [start]

        # Priority queue: (f_score, counter, position)
        # Counter breaks ties deterministically and avoids tuple comparison issues
        open_set = []
        counter = 0
        heapq.heappush(open_set, (self._heuristic(start, end), counter, start))

        g_score = {start: 0}
        came_from = {}
        closed_set = set()

        # 4-directional movement: right, left, down, up
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                curr = current
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.append(start)
                return path[::-1]

            if current in closed_set:
                continue

            closed_set.add(current)

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._is_in_bounds(neighbor) or self._is_wall(neighbor):
                    continue

                # Cost to enter the neighbor cell
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return None

import pytest

def test_basic_path():
    """Test standard pathfinding on an unweighted grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Optimal Manhattan path length

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_exists():
    """Test when destination is completely blocked by walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_start_end_raises_value_error():
    """Test that starting or ending on a wall raises ValueError."""
    grid = [[1, 0], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="on a wall"):
        astar.find_path((0, 1), (1, 1))

def test_weighted_optimal_path():
    """Test that A* chooses the lower-cost path over the shorter path."""
    grid = [
        [1, 100, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Direct path cost: 100 + 1 = 101
    # Detour path cost: 1 + 1 + 1 + 1 = 4
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    assert (1, 1) in path  # Must go through middle row
    assert (0, 1) not in path  # Must avoid expensive cell