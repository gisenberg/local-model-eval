from __future__ import annotations
import heapq
from typing import List, Tuple, Optional, Set, Dict

class AStarGrid:
    """
    A* pathfinding implementation for a weighted 2D grid.
    
    The grid is represented as a 2D list where:
    - 0 represents a wall (impassable cell)
    - Positive integers represent traversable cells with the value as the movement cost
    
    Methods:
        find_path(start, end): Finds the optimal path from start to end coordinates
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D grid.
        
        Args:
            grid: 2D list of integers where 0 represents walls and positive integers represent traversable cells
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end coordinates using A* algorithm.
        
        Args:
            start: Tuple of (row, col) coordinates for the starting position
            end: Tuple of (row, col) coordinates for the ending position
            
        Returns:
            List of coordinates representing the optimal path from start to end, 
            or None if no path exists
            
        Raises:
            ValueError: If start or end coordinates are out of bounds
        """
        # Validate input coordinates
        if not self._is_valid_position(start):
            raise ValueError(f"Start position {start} is out of bounds")
        if not self._is_valid_position(end):
            raise ValueError(f"End position {end} is out of bounds")
            
        # Handle special case where start and end are the same
        if start == end:
            return [start]
            
        # Initialize data structures
        open_set = []  # Priority queue (min-heap) of (f_score, g_score, position)
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}  # For path reconstruction
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        f_score: Dict[Tuple[int, int], float] = {start: self._heuristic(start, end)}
        
        closed_set: Set[Tuple[int, int]] = set()
        
        # Directions: (dr, dc) for up, right, down, left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        while open_set:
            # Get the cell with the lowest f_score
            _, current_g, current = heapq.heappop(open_set)
            
            # Check if we've reached the end
            if current == end:
                return self._reconstruct_path(came_from, current)
                
            # Skip if this position has already been evaluated
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            # Explore neighbors
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                # Skip if neighbor is out of bounds or a wall
                if not self._is_valid_position(neighbor) or self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue
                    
                # Calculate tentative g_score (cost from start to neighbor)
                tentative_g = current_g + self.grid[neighbor[0]][neighbor[1]]
                
                # If this path to neighbor is better than any previously examined
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Update path information
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)
                    
                    # Add to open set if not already present
                    if not any(neighbor == item[2] for item in open_set):
                        heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
        
        # No path found
        return None
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within grid bounds."""
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _heuristic(self, start: Tuple[int, int], end: Tuple[int, int]) -> float:
        """
        Calculate Manhattan distance heuristic.
        
        Args:
            start: Starting position (row, col)
            end: Ending position (row, col)
            
        Returns:
            Estimated cost from start to end (Manhattan distance)
        """
        return abs(start[0] - end[0]) + abs(start[1] - end[1])
    
    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to current position."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

import pytest
from typing import Optional, List, Tuple

def test_start_end_same():
    """Test that start and end positions are the same."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    finder = AStarGrid(grid)
    path = finder.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_exists():
    """Test when no path exists due to walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    finder = AStarGrid(grid)
    path = finder.find_path((0, 0), (2, 2))
    assert path is None

def test_out_of_bounds_start():
    """Test ValueError for out-of-bounds start position."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError, match="Start position"):
        finder.find_path((2, 0), (0, 0))

def test_out_of_bounds_end():
    """Test ValueError for out-of-bounds end position."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError, match="End position"):
        finder.find_path((0, 0), (2, 2))

def test_wall_at_start():
    """Test that wall at start position prevents path finding."""
    grid = [
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    finder = AStarGrid(grid)
    path = finder.find_path((0, 0), (2, 2))
    assert path is None

def test_weighted_path():
    """Test path with different movement costs."""
    grid = [
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ]
    finder = AStarGrid(grid)
    path = finder.find_path((0, 0), (2, 2))
    
    # The optimal path should go around the high-cost cell (5)
    # Expected path: [(0,0), (0,1), (0,2), (1,2), (2,2)]
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]