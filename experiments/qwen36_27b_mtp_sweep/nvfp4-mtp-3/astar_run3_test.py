import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    Cells with value 0 are walls (impassable).
    Positive integer values represent traversal costs/weights.
    """
    def __init__(self, grid: List[List[int]]):
        """Initialize the grid with a 2D list of weights."""
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid boundaries."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a wall (value 0)."""
        return self.grid[pos[0]][pos[1]] == 0

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.
        
        Returns:
            List of (row, col) tuples representing the optimal path,
            or None if no valid path exists.
            
        Raises:
            ValueError: If start/end are out of bounds or placed on walls.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds.")
        if self._is_wall(start) or self._is_wall(end):
            raise ValueError("Start or end position is a wall.")

        if start == end:
            return [start]

        # Priority queue stores tuples: (f_score, tie_breaker, position)
        open_set: list = []
        heapq.heappush(open_set, (0, 0, start))
        g_score: dict = {start: 0}
        came_from: dict = {}
        tie_breaker = 1

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_set:
            f, _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                return path[::-1]

            # Skip stale entries that have been superseded by better paths
            if f > g_score[current]:
                continue

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._is_valid(neighbor) or self._is_wall(neighbor):
                    continue

                weight = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + weight

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = self._manhattan(neighbor, end)
                    f_new = tentative_g + h
                    heapq.heappush(open_set, (f_new, tie_breaker, neighbor))
                    tie_breaker += 1

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
    assert len(path) == 5  # Optimal steps: 4 moves + start

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_no_path():
    """Test when destination is completely blocked by walls."""
    grid = [[1, 0, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None

def test_out_of_bounds():
    """Test ValueError for coordinates outside grid dimensions."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_start_is_wall():
    """Test ValueError when start or end position is a wall."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 1))  # Start is wall
    with pytest.raises(ValueError):
        astar.find_path((0, 1), (0, 0))  # End is wall

def test_weighted_optimal():
    """Test that A* chooses lowest-cost path over shortest-step path."""
    grid = [
        [1, 100, 1],
        [1,   1, 1],
        [1,   1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    assert (0, 1) not in path  # Must avoid the high-cost cell
    # Verify total cost matches the cheaper detour: 1+1+1+1 = 4
    cost = sum(grid[r][c] for r, c in path[1:])
    assert cost == 4