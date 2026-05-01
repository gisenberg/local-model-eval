from __future__ import annotations
import heapq
from typing import List, Tuple, Optional, Set, Dict

class AStarGrid:
    """
    A* pathfinding implementation for a weighted 2D grid.
    
    The grid uses 0 to represent walls (impassable cells) and positive integers for weighted cells.
    Only 4-directional movement (up, down, left, right) is allowed.
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D grid.
        
        Args:
            grid: 2D list of integers where 0 represents a wall and positive integers represent weighted cells.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: Starting position as (row, col) tuple.
            end: Ending position as (row, col) tuple.
            
        Returns:
            List of (row, col) tuples representing the path from start to end, or None if no path exists.
            
        Raises:
            ValueError: If start or end positions are out of bounds.
        """
        # Check if start and end positions are valid
        if not self._is_valid_position(start):
            raise ValueError(f"Start position {start} is out of bounds")
        if not self._is_valid_position(end):
            raise ValueError(f"End position {end} is out of bounds")
            
        # Handle special case where start and end are the same
        if start == end:
            return [start]
            
        # Initialize data structures
        open_set: Set[Tuple[int, int]] = set()
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {}
        f_score: Dict[Tuple[int, int], float] = {}
        
        # Set up initial values
        g_score[start] = 0
        f_score[start] = self._heuristic(start, end)
        
        # Priority queue: (f_score, position)
        heap = [(f_score[start], start)]
        open_set.add(start)
        
        while heap:
            current = heapq.heappop(heap)[1]
            
            # Check if we've reached the end
            if current == end:
                return self._reconstruct_path(came_from, start, end)
                
            open_set.remove(current)
            
            # Check all 4-directional neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Skip if neighbor is invalid
                if not self._is_valid_position(neighbor):
                    continue
                    
                # Skip if neighbor is a wall
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue
                    
                # Calculate tentative g_score
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]
                
                # If this path to neighbor is better, update
                if neighbor not in open_set or tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)
                    
                    if neighbor not in open_set:
                        open_set.add(neighbor)
                        heapq.heappush(heap, (f_score[neighbor], neighbor))
                        
        # No path found
        return None
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within grid bounds."""
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _heuristic(self, start: Tuple[int, int], end: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(start[0] - end[0]) + abs(start[1] - end[1])
    
    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], 
                          start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to end."""
        path = [end]
        current = end
        
        while current != start:
            current = came_from[current]
            path.append(current)
            
        path.reverse()
        return path

import pytest

def test_start_end_same():
    """Test when start and end positions are the same."""
    grid = [[1, 2], [3, 4]]
    a_star = AStarGrid(grid)
    path = a_star.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_wall_at_start():
    """Test when start position is a wall."""
    grid = [[0, 1], [1, 1]]
    a_star = AStarGrid(grid)
    with pytest.raises(ValueError):
        a_star.find_path((0, 0), (1, 1))

def test_wall_at_end():
    """Test when end position is a wall."""
    grid = [[1, 1], [1, 0]]
    a_star = AStarGrid(grid)
    with pytest.raises(ValueError):
        a_star.find_path((0, 0), (1, 1))

def test_no_path_exists():
    """Test when no path exists due to walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    a_star = AStarGrid(grid)
    path = a_star.find_path((0, 0), (2, 2))
    assert path is None

def test_simple_path():
    """Test a simple path with no obstacles."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    a_star = AStarGrid(grid)
    path = a_star.find_path((0, 0), (2, 2))
    assert len(path) == 5  # Should be 5 steps (Manhattan distance)
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

def test_weighted_path():
    """Test path with weighted cells."""
    grid = [
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ]
    a_star = AStarGrid(grid)
    path = a_star.find_path((0, 0), (2, 2))
    # The optimal path should go around the weighted cell (5)
    assert path[1] == (0, 1)  # Go right first instead of down
    assert len(path) == 5  # Still 5 steps (Manhattan distance)