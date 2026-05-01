from __future__ import annotations
import heapq
from typing import List, Tuple, Optional, Set, Dict

class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.
    
    Args:
        grid: 2D list of integers where 0 represents walls and positive integers represent walkable cells with their weights.
    
    Methods:
        find_path(start, end): Finds the optimal path from start to end coordinates.
    """
    
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end coordinates using A* algorithm.
        
        Args:
            start: Tuple of (row, col) coordinates for the starting position.
            end: Tuple of (row, col) coordinates for the ending position.
            
        Returns:
            List of coordinates representing the optimal path from start to end, 
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
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
        open_set: Set[Tuple[int, int]] = set()
        closed_set: Set[Tuple[int, int]] = set()
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        f_score: Dict[Tuple[int, int], float] = {start: self._heuristic(start, end)}
        
        # Priority queue: (f_score, g_score, position)
        heap = [(f_score[start], g_score[start], start)]
        
        while heap:
            _, current_g, current = heapq.heappop(heap)
            
            # Check if we've reached the end
            if current == end:
                return self._reconstruct_path(came_from, current)
                
            open_set.discard(current)
            closed_set.add(current)
            
            # Check all 4-directional neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Skip if neighbor is invalid
                if not self._is_valid_position(neighbor):
                    continue
                    
                # Skip if neighbor is a wall or already in closed set
                if self.grid[neighbor[0]][neighbor[1]] == 0 or neighbor in closed_set:
                    continue
                    
                # Calculate tentative g_score
                tentative_g = current_g + self.grid[neighbor[0]][neighbor[1]]
                
                # If this path to neighbor is better, update
                if neighbor not in open_set or tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)
                    
                    if neighbor not in open_set:
                        open_set.add(neighbor)
                        heapq.heappush(heap, (f_score[neighbor], tentative_g, neighbor))
                        
        # No path found
        return None
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within grid bounds."""
        row, col = pos
        return (0 <= row < self.rows and 
                0 <= col < self.cols and 
                self.grid[row][col] != 0)
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from end to start."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

import pytest
from typing import Optional, List, Tuple

def test_start_equal_end():
    """Test that start and end at same position returns path with single point."""
    grid = [[1, 1], [1, 1]]
    finder = AStarGrid(grid)
    path = finder.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_exists():
    """Test that no path exists when surrounded by walls."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    finder = AStarGrid(grid)
    path = finder.find_path((0, 0), (2, 2))
    assert path is None

def test_direct_path():
    """Test direct path with no obstacles."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    finder = AStarGrid(grid)
    path = finder.find_path((0, 0), (2, 2))
    # Should be direct diagonal path (3 steps)
    assert len(path) == 3
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

def test_weighted_path():
    """Test path with different weights affecting the route."""
    grid = [
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ]
    finder = AStarGrid(grid)
    path = finder.find_path((0, 0), (2, 2))
    # Should avoid the center cell with weight 5
    assert path[1] != (1, 1)

def test_out_of_bounds_start():
    """Test ValueError for out-of-bounds start position."""
    grid = [[1, 1], [1, 1]]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError):
        finder.find_path((2, 2), (0, 0))

def test_out_of_bounds_end():
    """Test ValueError for out-of-bounds end position."""
    grid = [[1, 1], [1, 1]]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError):
        finder.find_path((0, 0), (3, 3))

def test_wall_start():
    """Test ValueError when start position is a wall."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError):
        finder.find_path((0, 0), (1, 1))

def test_wall_end():
    """Test ValueError when end position is a wall."""
    grid = [
        [1, 1],
        [1, 0]
    ]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError):
        finder.find_path((0, 0), (1, 1))