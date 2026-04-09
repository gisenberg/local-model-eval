import heapq
from typing import List, Tuple, Optional, Dict


class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize A* pathfinder on a weighted 2D grid.
        
        Args:
            grid: 2D list where 0 = wall, positive int = movement cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.
        
        Returns:
            List of (row, col) coordinates from start to end inclusive, 
            or None if no path exists.
        
        Raises:
            ValueError: If start or end is out of bounds.
        """
        # 1. Bounds checking
        if not self._is_valid(*start) or not self._is_valid(*end):
            raise ValueError("Start or end coordinates out of bounds")

        # 2. Edge cases
        if start == end:
            return [start]
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # 3. Data structures
        # open_set: min-heap of (f_score, current_node)
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, end), start))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        
        # Track visited to avoid re-processing nodes with higher costs
        visited = set()

        # 4. Main loop
        while open_set:
            f_curr, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            if current in visited:
                continue
            visited.add(current)

            # 4-directional neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dr, current[1] + dc)

                if not self._is_valid(*neighbor):
                    continue
                
                cost = self.grid[neighbor[0]][neighbor[1]]
                if cost == 0:  # Wall
                    continue

                tentative_g = g_score[current] + cost
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))

        return None


# ==============================================================================
# Pytest Tests
# ==============================================================================

import pytest

def calculate_path_cost(grid, path):
    """Helper to calculate total cost of a path (excluding start cell)."""
    return sum(grid[r][c] for r, c in path[1:])

def test_simple_path():
    """Test pathfinding on a uniform grid with no obstacles."""
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    solver = AStarGrid(grid)
    path = solver.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5  # 0,0 -> 0,1 -> 0,2 -> 1,2 -> 2,2 (or equivalent)
    assert calculate_path_cost(grid, path) == 4

def test_around_obstacles():
    """Test pathfinding that must navigate around walls."""
    grid = [[1, 1, 1],
            [0, 0, 1],
            [1, 1, 1]]
    solver = AStarGrid(grid)
    path = solver.find_path((0, 0), (2, 0))
    # Must go around: (0,0)->(0,1)->(0,2)->(1,2)->(2,2)->(2,1)->(2,0)
    assert path is not None
    assert len(path) == 7
    assert calculate_path_cost(grid, path) == 6

def test_weighted_grid():
    """Test that path prefers lower-cost cells over shorter geometric paths."""
    # Direct path (0,0)->(1,0)->(2,0) costs 10+10=20
    # Detour (0,0)->(0,1)->(1,1)->(2,1)->(2,0) costs 1+1+1+1=4
    grid = [[1, 1, 1],
            [10, 1, 1],
            [10, 1, 1]]
    solver = AStarGrid(grid)
    path = solver.find_path((0, 0), (2, 0))
    assert path is not None
    assert calculate_path_cost(grid, path) == 4
    assert (1, 0) not in path  # Should avoid the expensive cell

def test_no_path():
    """Test scenario where start and end are separated by walls."""
    grid = [[1, 0, 1],
            [1, 0, 1],
            [1, 0, 1]]
    solver = AStarGrid(grid)
    path = solver.find_path((0, 0), (0, 2))
    assert path is None

def test_start_equals_end():
    """Test that start == end returns a single-element list."""
    grid = [[1, 1],
            [1, 1]]
    solver = AStarGrid(grid)
    path = solver.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_invalid_coordinates():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1],
            [1, 1]]
    solver = AStarGrid(grid)
    with pytest.raises(ValueError):
        solver.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        solver.find_path((0, 0), (5, 5))