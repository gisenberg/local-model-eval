import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.

    Grid conventions:
    - 0 represents a wall (impassable)
    - Positive integers represent traversable cells with traversal cost equal to their value
    """

    def __init__(self, grid: List[List[int]]):
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start: (row, col) tuple of starting position
            end: (row, col) tuple of target position

        Returns:
            List of (row, col) tuples representing the optimal path, or None if no path exists.

        Raises:
            ValueError: If start or end is out of bounds or on a wall.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds or on a wall.")

        if start == end:
            return [start]

        # Priority queue stores (f_score, counter, position)
        # Counter ensures stable sorting and avoids tuple comparison errors
        counter = 0
        open_set = [(0, counter, start)]
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        visited = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current in visited:
                continue
            visited.add(current)

            if current == end:
                return self._reconstruct_path(came_from, current)

            # 4-directional movement: right, left, down, up
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._is_valid(neighbor):
                    continue

                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return None

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not a wall."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic for 4-directional movement."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Backtrack from end to start using came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

import pytest

def test_simple_open_grid():
    """Test basic pathfinding on an unweighted grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Verify optimal cost (4 steps of cost 1)
    cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert cost == 4

def test_walls_avoided():
    """Test that path correctly navigates around walls."""
    grid = [[1, 0, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert (0, 1) not in path
    assert (1, 1) not in path

def test_weighted_optimal_path():
    """Test that A* chooses lower-weight routes over shorter geometric paths."""
    grid = [
        [1, 10, 1],
        [1, 1, 1],
        [1, 10, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    # Optimal path goes through middle row to avoid weight 10 cells
    assert (0, 1) not in path
    assert (2, 1) not in path
    # Verify optimal cost (4 steps of cost 1)
    cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert cost == 4

def test_start_equals_end():
    """Test immediate return when start and end are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_out_of_bounds_raises_value_error():
    """Test that invalid coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_no_path_returns_none():
    """Test that enclosed start position returns None."""
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (0, 0))
    assert path is None