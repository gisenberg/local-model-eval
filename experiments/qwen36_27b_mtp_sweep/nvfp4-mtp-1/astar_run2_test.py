import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid semantics:
    - 0: Wall (impassable)
    - Positive integers: Movement cost to enter the cell
    - Movement: 4-directional (up, down, left, right)
    - Heuristic: Manhattan distance (consistent, guarantees optimality)
    """
    
    def __init__(self, grid: List[List[int]]):
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, r: int, c: int) -> bool:
        """Check if a cell is a wall."""
        return self.grid[r][c] == 0

    def _heuristic(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """Manhattan distance heuristic for 4-directional movement."""
        return abs(r1 - r2) + abs(c1 - c2)

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end.
        
        Args:
            start: (row, col) tuple of the starting position
            end: (row, col) tuple of the target position
            
        Returns:
            List of (row, col) tuples representing the optimal path, or None if unreachable.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        sr, sc = start
        er, ec = end

        if not self._is_valid(sr, sc) or not self._is_valid(er, ec):
            raise ValueError("Start or end position is out of bounds")

        if self._is_wall(sr, sc) or self._is_wall(er, ec):
            return None

        if start == end:
            return [start]

        # Priority queue: (f_score, counter, (r, c))
        # Counter ensures FIFO ordering for equal f_scores, avoiding tuple comparison errors
        counter = 0
        open_set = [(self._heuristic(sr, sc, er, ec), counter, (sr, sc))]
        g_score = {(sr, sc): 0}
        came_from = {}

        while open_set:
            f, _, current = heapq.heappop(open_set)
            cr, cc = current

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Explore 4-directional neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = cr + dr, cc + dc
                if not self._is_valid(nr, nc) or self._is_wall(nr, nc):
                    continue

                move_cost = self.grid[nr][nc]
                tentative_g = g_score[current] + move_cost

                # Found a better path to neighbor
                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + self._heuristic(nr, nc, er, ec)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, (nr, nc)))

        return None

import pytest
from typing import List, Tuple

# Import AStarGrid from your module
# 
def _calculate_path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Helper to verify path cost matches grid weights."""
    cost = 0
    for i in range(1, len(path)):
        r, c = path[i]
        cost += grid[r][c]
    return cost

def test_basic_path():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (2, 2)
    assert len(path) == 5  # 4 steps + start
    assert _calculate_path_cost(grid, path) == 4

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_out_of_bounds():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_wall_blockage():
    # Column 1 is completely walled off
    grid = [[1, 0, 1], [1, 0, 1], [1, 0, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_weighted_optimal():
    # Center cell is expensive (9), path should route around it
    grid = [
        [1, 1, 1],
        [1, 9, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Optimal cost is 4 (going around), not 11+ (going through center)
    assert _calculate_path_cost(grid, path) == 4

def test_start_or_end_is_wall():
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (1, 1)) is None
    assert astar.find_path((1, 0), (0, 0)) is None