import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid representation:
    - 0: Wall (unwalkable)
    - Positive integers: Cost to enter/move through the cell
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid for pathfinding.
        
        Args:
            grid: 2D list of integers representing cell weights. 
                  0 denotes a wall, >0 denotes traversal cost.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within grid boundaries."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, r: int, c: int) -> bool:
        """Check if a cell is a wall."""
        return self.grid[r][c] == 0

    def _heuristic(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """Manhattan distance heuristic for 4-directional movement."""
        return abs(r1 - r2) + abs(c1 - c2)

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.
        
        Args:
            start: (row, col) tuple of the starting position.
            end: (row, col) tuple of the target position.
            
        Returns:
            List of (row, col) tuples representing the optimal path, 
            or None if no valid path exists.
            
        Raises:
            ValueError: If start/end are out of bounds or located on walls.
        """
        sr, sc = start
        er, ec = end

        if not (self._is_valid(sr, sc) and self._is_valid(er, ec)):
            raise ValueError("Start or end position is out of bounds")
            
        if self._is_wall(sr, sc) or self._is_wall(er, ec):
            raise ValueError("Start or end position is a wall")

        if start == end:
            return [start]

        # Priority queue: (f_score, g_score, counter, (r, c))
        # counter ensures FIFO ordering for equal f-scores
        open_set = [(self._heuristic(sr, sc, er, ec), 0, 0, (sr, sc))]
        counter = 1

        g_score = {start: 0}
        came_from = {}

        while open_set:
            f, g, _, current = heapq.heappop(open_set)
            cr, cc = current

            # Skip stale entries where a better path was already found
            if g > g_score.get(current, float('inf')):
                continue

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

                # Cost to enter the neighbor cell
                tentative_g = g + self.grid[nr][nc]

                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + self._heuristic(nr, nc, er, ec)
                    heapq.heappush(open_set, (f_score, tentative_g, counter, (nr, nc)))
                    counter += 1

        return None

import pytest
from typing import List, Tuple, Optional

# Import AStarGrid from your module
# 
def test_simple_direct_path() -> None:
    """Test basic pathfinding on an open grid."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

def test_start_equals_end() -> None:
    """Test when start and end positions are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_exists() -> None:
    """Test when walls completely block the target."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds_raises_value_error() -> None:
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))

def test_wall_at_start_or_end_raises_value_error() -> None:
    """Test that starting or ending on a wall raises ValueError."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((1, 1), (0, 0))

def test_weighted_cells_optimal_path() -> None:
    """Test that A* correctly chooses lower-weight paths over shorter geometric paths."""
    # Direct path cost: 10 + 10 + 1 = 21
    # Detour path cost: 1 + 1 + 1 + 1 + 1 = 5
    grid = [
        [1, 10, 10, 1],
        [1,  1,  1, 1],
        [1, 10, 10, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 3))
    assert path is not None
    # Optimal path should detour through row 1
    expected = [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (0, 3)]
    assert path == expected