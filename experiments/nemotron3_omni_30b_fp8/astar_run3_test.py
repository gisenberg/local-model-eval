from __future__ import annotations
import heapq
from typing import List, Tuple, Optional, Set, Dict

class AStarGrid:
    """
    A* pathfinding implementation for a weighted 2D grid.
    
    The grid uses 0 to represent walls (impassable cells) and positive integers 
    to represent walkable cells with their respective movement costs.
    
    Methods:
        find_path(start, end): Finds the optimal path from start to end coordinates.
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D grid.
        
        Args:
            grid: 2D list of integers where 0 represents walls and positive integers 
                  represent walkable cells with their movement costs.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end coordinates using A* algorithm.
        
        Args:
            start: Tuple of (row, col) coordinates for the starting position.
            end: Tuple of (row, col) coordinates for the destination position.
            
        Returns:
            List of (row, col) coordinates representing the optimal path from start to end,
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of grid bounds.
        """
        # Validate input coordinates
        if not self._is_valid_position(start):
            raise ValueError(f"Start position {start} is out of bounds")
        if not self._is_valid_position(end):
            raise ValueError(f"End position {end} is out of bounds")
            
        # Handle start == end case
        if start == end:
            return [start]
            
        # Initialize data structures
        open_set = []  # Priority queue (min-heap) of (f_score, g_score, position)
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}  # For path reconstruction
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        f_score: Dict[Tuple[int, int], float] = {start: self._heuristic(start, end)}
        
        closed_set: Set[Tuple[int, int]] = set()
        
        # 4-directional movement (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while open_set:
            _, current_g, current_pos = heapq.heappop(open_set)
            
            # Goal check
            if current_pos == end:
                return self._reconstruct_path(came_from, current_pos)
                
            closed_set.add(current_pos)
            
            # Check neighbors
            for dx, dy in directions:
                new_pos = (current_pos[0] + dx, current_pos[1] + dy)
                
                # Skip if out of bounds or wall
                if not self._is_valid_position(new_pos) or self.grid[new_pos[0]][new_pos[1]] == 0:
                    continue
                    
                # Skip if already in closed set
                if new_pos in closed_set:
                    continue
                    
                # Calculate tentative g_score
                tentative_g = current_g + self.grid[new_pos[0]][new_pos[1]]
                
                # If this path is better, record it
                if new_pos not in g_score or tentative_g < g_score[new_pos]:
                    came_from[new_pos] = current_pos
                    g_score[new_pos] = tentative_g
                    f = tentative_g + self._heuristic(new_pos, end)
                    f_score[new_pos] = f
                    heapq.heappush(open_set, (f, tentative_g, new_pos))
                    
        return None  # No path found
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate Manhattan distance heuristic.
        
        Args:
            pos1: First position (row, col)
            pos2: Second position (row, col)
            
        Returns:
            Manhattan distance between pos1 and pos2
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from start to current position."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get start -> end order

import pytest
from typing import List, Tuple, Optional

def test_start_equal_end():
    """Test that start == end returns path with single point."""
    grid = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    finder = AStarGrid(grid)
    path = finder.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_exists():
    """Test when no path exists due to walls."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError):
        finder.find_path((0, 0), (2, 2))

def test_out_of_bounds_start():
    """Test ValueError for out-of-bounds start position."""
    grid = [
        [1, 2],
        [3, 4]
    ]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError):
        finder.find_path((2, 2), (0, 0))

def test_out_of_bounds_end():
    """Test ValueError for out-of-bounds end position."""
    grid = [
        [1, 2],
        [3, 4]
    ]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError):
        finder.find_path((0, 0), (2, 2))

def test_wall_at_start():
    """Test ValueError when start position is a wall."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError):
        finder.find_path((0, 0), (1, 1))

def test_wall_at_end():
    """Test ValueError when end position is a wall."""
    grid = [
        [1, 1],
        [1, 0]
    ]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError):
        finder.find_path((0, 0), (1, 1))

def test_optimal_path():
    """Test that path is optimal (shortest total cost)."""
    grid = [
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ]
    finder = AStarGrid(grid)
    path = finder.find_path((0, 0), (2, 2))
    
    # Expected path: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) (cost = 1+1+1+1+1 = 5)
    # Alternative path through center: (0,0) -> (1,0) -> (1,1) -> (2,1) -> (2,2) (cost = 1+1+5+1+1 = 9)
    expected_path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
    assert path == expected_path