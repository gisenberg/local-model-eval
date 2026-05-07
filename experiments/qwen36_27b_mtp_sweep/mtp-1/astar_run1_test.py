import heapq
from typing import List, Tuple, Optional, Dict, Set
from itertools import count


class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid representation:
    - 0: Wall (impassable)
    - Positive integers: Movement cost to enter the cell
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the AStarGrid.
        
        Args:
            grid: 2D list of integers. Must be rectangular and non-empty.
                  0 represents walls, >0 represents movement costs.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty and rectangular.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        # Tie-breaker counter to prevent tuple comparison errors in heapq
        self._counter = count()

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* search.
        
        Args:
            start: (row, col) tuple for the starting position.
            end: (row, col) tuple for the target position.
            
        Returns:
            List of (row, col) tuples representing the optimal path from start to end,
            inclusive. Returns None if no valid path exists.
            
        Raises:
            ValueError: If start/end are out of bounds or positioned on a wall.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds.")
            
        # Validate walls
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall.")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall.")
            
        # Trivial case
        if start == end:
            return [start]
            
        # Priority queue: (f_score, tie_breaker, (row, col))
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, next(self._counter), start))
        
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set: Set[Tuple[int, int]] = set()
        
        # 4-directional movement
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current == end:
                # Reconstruct path
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                return path[::-1]
                
            if current in closed_set:
                continue
            closed_set.add(current)
            
            cr, cc = current
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                
                # Bounds check
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                # Wall check
                if self.grid[nr][nc] == 0:
                    continue
                # Skip already finalized nodes
                if (nr, nc) in closed_set:
                    continue
                    
                move_cost = self.grid[nr][nc]
                tentative_g = g_score[current] + move_cost
                
                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    # Manhattan heuristic (admissible & consistent for 4-dir)
                    h_score = abs(nr - end[0]) + abs(nc - end[1])
                    f_score = tentative_g + h_score
                    heapq.heappush(open_set, (f_score, next(self._counter), (nr, nc)))
                    
        return None

import pytest

def test_basic_pathfinding():
    """Test standard path around a wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Must route around the center wall
    assert (1, 1) not in path

def test_start_equals_end():
    """Test trivial case where start and end are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_exists():
    """Test when end is completely blocked by walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds_raises_valueerror():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_at_start_or_end_raises_valueerror():
    """Test that starting or ending on a wall raises ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))
        
    grid2 = [[1, 1], [1, 0]]
    astar2 = AStarGrid(grid2)
    with pytest.raises(ValueError, match="wall"):
        astar2.find_path((0, 0), (1, 1))

def test_weighted_optimal_path():
    """Test that A* chooses lower cost path over shorter step count."""
    # Direct path costs 10, detour costs 1+1+1=3
    grid = [
        [1, 10, 1],
        [1,  1, 1],
        [1,  1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    
    # Verify it avoids the expensive cell
    assert (0, 1) not in path
    
    # Verify total cost (sum of weights for all cells except start)
    path_cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert path_cost == 3