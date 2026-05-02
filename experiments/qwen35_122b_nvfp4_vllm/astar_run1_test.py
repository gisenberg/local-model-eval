import heapq
from typing import List, Tuple, Optional, Dict, Set
from math import inf

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid pathfinding object.
        
        Args:
            grid: 2D list representing the maze. 
                  0 represents a wall. Positive integers represent movement cost.
        """
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if grid else 0
        
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using the A* algorithm.
        
        Uses a 4-directional movement model, Manhattan heuristic, and respects 
        cell weights (movement costs). Returns None if no path exists.
        
        Args:
            start: Tuple (row, col) of the starting position.
            end: Tuple (row, col) of the destination position.
            
        Returns:
            List of coordinates representing the path, or None if unreachable.
            Raises ValueError if coordinates are out of bounds or land on a wall.
            
        Examples:
            >>> grid = [[1, 1], [1, 1]]
            >>> astar = AStarGrid(grid)
            >>> astar.find_path((0,0), (1,1))
            [(0, 0), (0, 1), (1, 1)] # or similar optimal path
        """
        # 1. Validation: Out of bounds
        sr, sc = start
        er, ec = end
        
        if not (0 <= sr < self.height and 0 <= sc < self.width):
            raise ValueError(f"Start position {start} is out of bounds.")
        if not (0 <= er < self.height and 0 <= ec < self.width):
            raise ValueError(f"End position {end} is out of bounds.")
            
        # 2. Validation: Start/End on Wall
        if self.grid[sr][sc] == 0:
            raise ValueError(f"Start position {start} is on a wall (0).")
        if self.grid[er][ec] == 0:
            raise ValueError(f"End position {end} is on a wall (0).")
            
        # 3. Special Case: Start == End
        if start == end:
            return [start]
            
        # A* Data Structures
        open_set: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, 0.0, start))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score: current best cost to reach node from start
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        
        # f_score: g_score + heuristic
        f_score: Dict[Tuple[int, int], float] = {start: self._heuristic(start, end)}
        
        closed_set: Set[Tuple[int, int]] = set()
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            # Goal Reached
            if current == end:
                return self._reconstruct_path(came_from, end)
            
            # Avoid revisiting processed nodes
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            curr_r, curr_c = current
            
            # Neighbors: 4-directional
            directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            
            for dr, dc in directions:
                nr, nc = curr_r + dr, curr_c + dc
                
                # Boundary Check (Neighbors)
                if not (0 <= nr < self.height and 0 <= nc < self.width):
                    continue
                    
                # Wall Check
                weight = self.grid[nr][nc]
                if weight == 0:
                    continue
                
                neighbor = (nr, nc)
                
                tentative_g = g_score[current] + weight
                
                # Relaxation: Found a better path to neighbor
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    f_score[neighbor] = f
                    
                    # Push to priority queue: (f, g, node)
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
                    
        # No path found
        return None

    def _heuristic(self, node: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Calculate Manhattan distance between two nodes.
        Assumes minimum movement cost of 1, making this heuristic admissible.
        """
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Backtracks from current node to start to build the final path."""
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path


# --- Pytest Tests ---
"""
To run these tests, save the above code in `astar.py` and the following in `test_astar.py`.
Install pytest: pip install pytest
Run: pytest test_astar.py -v
"""

import pytest
from unittest.mock import patch

def test_basic_unweighted_path():
    """Test 1: Basic path finding on a grid of 1s (equal weights)."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Length should be reasonable (Manhattan distance + 1)
    assert len(path) == 5 

def test_start_equals_end():
    """Test 2: Handling start coordinate equal to end coordinate."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 1), (0, 1))
    assert path is not None
    assert path == [(0, 1)]

def test_blocked_path_returns_none():
    """Test 3: Returns None when walls surround the destination."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    # Surround center with 0s relative to start (0,0) to (2,2) via center is blocked, 
    # but edges work. Let's block ALL paths.
    grid_blocked = [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1]
    ]
    # Actually simpler: 
    grid_simple_block = [
        [1, 0],
        [0, 1]
    ]
    astar = AStarGrid(grid_simple_block)
    path = astar.find_path((0, 0), (1, 1))
    assert path is None

def test_out_of_bounds_raises_value_error():
    """Test 4: Raises ValueError for coordinates outside grid dimensions."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (0, 1))
        
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (10, 10))

def test_optimal_weighted_path():
    """Test 5: Verifies A* finds optimal path on weighted grid."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    # Modify middle to make direct route expensive
    grid[1][1] = 100
    astar = AStarGrid(grid)
    # Path (0,0) -> (2,2)
    # Direct would cut corner: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) or similar
    # Cost calculation depends on entry.
    # Route A: Down, Down, Right, Right. Enter (1,0)[1], (2,0)[1], (2,1)[1], (2,2)[1]. Total 4.
    # Route B: Diagonal style through center? (0,0)->(1,0)->(1,1)[100]->... Cost 100+.
    # Should avoid center.
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert (1, 1) not in path, "Path should avoid the heavy cost cell"

def test_invalid_start_on_wall():
    """Test 6: Raises ValueError if start or end position is a wall (0)."""
    grid = [
        [1, 0],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError, match="on a wall"):
        astar.find_path((0, 1), (1, 1)) # Start is wall
        
    with pytest.raises(ValueError, match="on a wall"):
        astar.find_path((0, 0), (0, 1)) # End is wall