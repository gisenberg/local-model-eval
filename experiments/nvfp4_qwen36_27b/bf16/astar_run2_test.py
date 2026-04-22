import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding implementation on a 2D weighted grid.
    
    Grid conventions:
    - 0 represents an impassable wall.
    - Positive numbers represent traversal costs (weights) for entering a cell.
    - Weights must be non-negative for A* optimality guarantees.
    """
    
    def __init__(self, grid: List[List[float]]):
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if coordinates are within grid boundaries."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic for 4-directional movement."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using the A* algorithm.
        
        Args:
            start: Starting coordinates (row, col)
            end: Target coordinates (row, col)
            
        Returns:
            A list of coordinates from start to end, or None if no path exists.
            
        Raises:
            ValueError: If start/end are out of bounds or located on a wall.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds")
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            raise ValueError("Start or end position is a wall")

        if start == end:
            return [start]

        # Priority queue stores (f_score, tie_breaker_counter, position)
        # Counter prevents Python from comparing coordinate tuples when f_scores are equal
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))

        g_score = {start: 0}
        came_from = {}
        counter = 1

        while open_set:
            f, _, current = heapq.heappop(open_set)

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # 4-directional neighbors: right, left, down, up
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)

                if not self._is_valid(neighbor):
                    continue
                if self.grid[nr][nc] == 0:
                    continue

                # Cost to move to neighbor is the weight of the neighbor cell
                tentative_g = g_score[current] + self.grid[nr][nc]

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1

        return None

import pytest

def test_basic_pathfinding():
    """Test standard pathfinding on a uniform grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5
    assert path[0] == (0, 0) and path[-1] == (2, 2)

def test_start_equals_end():
    """Test when start and end coordinates are identical."""
    grid = [[5]]
    assert AStarGrid(grid).find_path((0, 0), (0, 0)) == [(0, 0)]

def test_no_path_exists():
    """Test grid where target is completely blocked by walls."""
    grid = [[1, 0, 1], [1, 0, 1], [1, 1, 1]]
    assert AStarGrid(grid).find_path((0, 0), (0, 2)) is None

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    with pytest.raises(ValueError, match="out of bounds"):
        AStarGrid(grid).find_path((0, 0), (2, 2))

def test_wall_at_start_or_end_raises_value_error():
    """Test that starting or ending on a wall raises ValueError."""
    grid = [[0, 1], [1, 1]]
    with pytest.raises(ValueError, match="wall"):
        AStarGrid(grid).find_path((0, 0), (0, 1))

def test_weighted_optimal_path():
    """Test that A* correctly avoids high-weight cells when a cheaper alternative exists."""
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    # Optimal path goes down the left column to avoid the 10-weight cells
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]