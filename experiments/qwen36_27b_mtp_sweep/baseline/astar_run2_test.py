import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid conventions:
    - 0 represents walls (impassable)
    - Positive integers represent the cost to enter that cell
    - Movement is 4-directional (up, down, left, right)
    - Heuristic: Manhattan distance (admissible for weights >= 1)
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the grid. Assumes rectangular grid with non-negative integers."""
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: (row, col) tuple for starting position
            end: (row, col) tuple for target position
            
        Returns:
            List of (row, col) tuples representing the optimal path, or None if unreachable.
            
        Raises:
            ValueError: If start/end are out of bounds or located on a wall.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds")
            
        # Validate walls
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall")
            
        # Trivial case
        if start == end:
            return [start]
            
        # Priority queue: (f_score, counter, (row, col))
        # Counter breaks ties deterministically and avoids tuple comparison errors
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))
        
        g_score: dict[Tuple[int, int], float] = {start: 0}
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        counter = 1
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while open_set:
            f, _, current = heapq.heappop(open_set)
            
            if current == end:
                # Reconstruct path by backtracking
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
                
            cr, cc = current
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                
                # Check bounds and walls
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != 0:
                    tentative_g = g_score[current] + self.grid[nr][nc]
                    
                    if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                        came_from[(nr, nc)] = current
                        g_score[(nr, nc)] = tentative_g
                        
                        # Manhattan heuristic
                        h = abs(nr - end[0]) + abs(nc - end[1])
                        f_score = tentative_g + h
                        
                        heapq.heappush(open_set, (f_score, counter, (nr, nc)))
                        counter += 1
                        
        return None

import pytest

def test_simple_path():
    """Basic pathfinding on an open grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5  # Optimal length for 3x3 grid
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

def test_path_around_wall():
    """Path correctly navigates around an obstacle."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 7  # Must detour around center
    assert (1, 1) not in path  # Wall not in path

def test_weighted_optimality():
    """A* chooses lower total weight over shorter geometric distance."""
    grid = [
        [1, 10, 10, 10, 10],
        [1,  0,  0,  0,  0],
        [1,  0,  0,  0,  0],
        [1,  1,  1,  1,  1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (3, 4))
    assert path is not None
    # Forces path along left/bottom edge (cost 7) instead of top edge (cost 50)
    expected = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)]
    assert path == expected

def test_start_equals_end():
    """Returns single-element list when start and end are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_out_of_bounds_value_error():
    """Raises ValueError for coordinates outside grid dimensions."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_at_start_end_value_error():
    """Raises ValueError when start or end is placed on a wall (0)."""
    grid_start_wall = [[0, 1], [1, 1]]
    astar1 = AStarGrid(grid_start_wall)
    with pytest.raises(ValueError, match="wall"):
        astar1.find_path((0, 0), (1, 1))
        
    grid_end_wall = [[1, 1], [1, 0]]
    astar2 = AStarGrid(grid_end_wall)
    with pytest.raises(ValueError, match="wall"):
        astar2.find_path((0, 0), (1, 1))