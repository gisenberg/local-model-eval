# astar_grid.py

import heapq
from typing import List, Tuple, Optional
import math

class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.
    
    The grid uses 0 to represent walls (impassable), and positive integers (≥1) for movement costs.
    Lower costs indicate easier terrain to traverse. The movement is 4-directional (no diagonals).
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid with the given cost grid.
        
        :param grid: 2D list of non-negative integers where 0 = wall and >0 = movement cost.
        """
        if not grid or any(len(row) == 0 for row in grid):
            raise ValueError("Grid must be non-empty with non-empty rows.")
        if any(len(row) != len(grid[0]) for row in grid):
            raise ValueError("All rows in grid must have the same length.")
        
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    def _in_bounds(self, r: int, c: int) -> bool:
        """Check if position (r, c) is within the grid boundaries."""
        return 0 <= r < self.rows and 0 <= c < self.cols
    
    def _is_passable(self, r: int, c: int) -> bool:
        """Check if cell (r, c) is passable (not a wall)."""
        return self.grid[r][c] > 0
    
    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Manhattan heuristic: |x1 - x2| + |y1 - y2|.
        
        Note: Since movement cost can be >1, this heuristic may be *inadmissible* in weighted cases.
        For optimality with arbitrary weights, a better heuristic is needed (e.g., min edge cost * Manhattan).
        However, we assume the heuristic remains effective for typical uniform or mildly varying weights.
        The implementation uses Manhattan for simplicity and correctness under common assumptions.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from `start` to `end` using A*.
        
        :param start: Starting cell as (row, col)
        :param end: Goal cell as (row, col)
        :return: List of cells from start to end (inclusive), or None if no path exists.
        :raises ValueError: If start/end are out of bounds.
        
        Uses heapq for the priority queue and implements standard A* with 4-directional movement.
        """
        # Validate inputs
        if not self._in_bounds(*start):
            raise ValueError(f"Start position {start} is out of bounds.")
        if not self._in_bounds(*end):
            raise ValueError(f"End position {end} is out of bounds.")
        if not self._is_passable(*start):
            raise ValueError(f"Start position {start} is a wall (cost=0).")
        if not self._is_passable(*end):
            raise ValueError(f"End position {end} is a wall (cost=0).")
        
        # Handle trivial case: start == end
        if start == end:
            return [start]
        
        # Priority queue: (f_score, g_score, (row, col))
        # g_score: cost from start to current node
        # f_score = g_score + h_score (heuristic estimate to goal)
        open_heap = []
        heapq.heappush(open_heap, (0, 0, start))
        
        # Track visited nodes and their best g_score
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: dict[Tuple[int, int], float] = {start: 0.0}
        
        # For reconstruction: map node -> its f_score for fast re-prioritization (if needed)
        # But we use a simple approach: allow duplicates in heap, but skip outdated ones.
        while open_heap:
            f_current, g_current, current = heapq.heappop(open_heap)
            
            # Skip if we've found a better path to current already
            if g_current > g_score.get(current, float('inf')):
                continue
            
            # Goal reached?
            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]  # reverse to get start -> end
            
            # Explore neighbors
            cr, cc = current
            for dr, dc in self.directions:
                nr, nc = cr + dr, cc + dc
                
                if not self._in_bounds(nr, nc) or not self._is_passable(nr, nc):
                    continue
                
                # Movement cost: cost of entering the neighbor cell
                move_cost = self.grid[nr][nc]
                tentative_g = g_current + move_cost
                
                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = (cr, cc)
                    g_score[(nr, nc)] = tentative_g
                    h = self._manhattan_distance((nr, nc), end)
                    f = tentative_g + h
                    heapq.heappush(open_heap, (f, tentative_g, (nr, nc)))
        
        # No path found
        return None

# test_astar_grid.py

import pytest

def test_start_equals_end():
    """Test case where start and end are the same cell."""
    grid = [[1, 2], [3, 4]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_simple_path():
    """Test path in a simple open grid."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],  # wall in middle
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Path must go around the wall
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Verify path is valid and minimal (Manhattan distance 4, but detour adds 2)
    # Minimum possible: 0→0 to 2→2 requires 4 moves, but with wall must detour → 6 steps total?
    # Actually: (0,0) → (1,0) → (2,0) → (2,1) → (2,2): 4 steps, cost = 1+1+1+1 = 4
    # Check length (number of cells)
    assert len(path) == 5  # start + 4 moves = 5 cells
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

def test_no_path_exists():
    """Test when end is unreachable (surrounded by walls)."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 0]  # goal is wall (but this will raise ValueError first)
    ]
    astar = AStarGrid(grid)
    # Make goal passable but blocked off
    grid = [
        [1, 0, 0],
        [1, 0, 0],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Start at (0,0), end at (2,2) — blocked by walls
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_weighted_path_optimal():
    """Test that A* picks the cheapest path in weighted grid."""
    grid = [
        [1, 1, 10],
        [1, 1, 10],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Path (0,0) → (1,0) → (2,0) → (2,1) → (2,2): cost = 1+1+1+1 = 4
    # Path (0,0) → (0,1) → (0,2) → (1,2) → (2,2): cost = 1+10+10+10+1 = 32 ❌
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5  # should choose cheapest, 4 moves → 5 cells
    # Verify total cost = 4
    total_cost = sum(grid[r][c] for r, c in path[1:])  # exclude start, cost of entering each next cell
    assert total_cost == 4

def test_invalid_positions():
    """Test ValueError for out-of-bounds or wall start/end."""
    grid = [[1, 2], [3, 0]]
    astar = AStarGrid(grid)
    
    # Out-of-bounds start
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))
    
    # Out-of-bounds end
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 0))
    
    # Start on wall (cost=0)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((1, 1), (0, 0))
    
    # End on wall
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))

def test_invalid_grid():
    """Test ValueError for malformed grid inputs."""
    # Empty grid
    with pytest.raises(ValueError, match="non-empty"):
        AStarGrid([])
    
    # Empty row
    with pytest.raises(ValueError, match="non-empty"):
        AStarGrid([[]])
    
    # Ragged rows
    with pytest.raises(ValueError, match="same length"):
        AStarGrid([[1, 2], [3]])