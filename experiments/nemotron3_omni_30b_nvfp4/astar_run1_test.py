from __future__ import annotations
import heapq
from typing import List, Tuple, Optional, Set, Dict

class AStarGrid:
    """
    A* pathfinding implementation for a weighted 2D grid.
    
    Args:
        grid: 2D list of integers where 0 represents walls and positive integers represent weighted cells
        start: Tuple of (row, col) coordinates for the starting position
        end: Tuple of (row, col) coordinates for the ending position
    
    Methods:
        find_path(start, end) -> Optional[List[Tuple[int,int]]]:
            Finds the optimal path from start to end using A* algorithm
    """
    
    def __init__(self, grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> None:
        self.grid = grid
        self.start = start
        self.end = end
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        
        # Validate start and end positions
        if not self._is_valid_position(start):
            raise ValueError(f"Start position {start} is out of bounds")
        if not self._is_valid_position(end):
            raise ValueError(f"End position {end} is out of bounds")
            
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError(f"Start position {start} is a wall")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError(f"End position {end} is a wall")
    
    def find_path(self) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using A* algorithm.
        
        Returns:
            List of (row, col) tuples representing the path from start to end,
            or None if no path exists
        """
        if self.start == self.end:
            return [self.start]
            
        open_set = []
        heapq.heappush(open_set, self._heuristic(self.start))
        heapq.heappush(open_set, (self._heuristic(self.start), self.start))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {self.start: 0}
        closed_set: Set[Tuple[int, int]] = set()
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-directional movement
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == self.end:
                return self._reconstruct_path(came_from, current)
                
            closed_set.add(current)
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not self._is_valid_position(neighbor) or (neighbor in closed_set):
                    continue
                    
                if self.grid[neighbor[0]][neighbor[1]] == 0:  # Skip walls
                    continue
                    
                tentative_g_score = g_score[current] + self.grid[neighbor[0]][neighbor[1]]
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    priority = tentative_g_score + self._heuristic(neighbor)
                    heapq.heappush(open_set, (priority, neighbor))
        
        return None  # No path found
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within grid bounds."""
        rows, cols = self.rows, self.cols
        return 0 <= pos[0] < rows and 0 <= pos[1] < cols
    
    def _heuristic(self, pos: Tuple[int, int]) -> float:
        """
        Manhattan distance heuristic.
        
        Args:
            pos: Current position (row, col)
            
        Returns:
            Estimated cost to reach end from current position
        """
        return abs(pos[0] - self.end[0]) + abs(pos[1] - self.end[1])
    
    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from start to current position."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

import pytest
from typing import List, Tuple, Optional

def test_start_end_same():
    grid = [[1, 1], [1, 1]]
    a_star = AStarGrid(grid, (0, 0), (0, 0))
    assert a_star.find_path() == [(0, 0)]

def test_wall_start():
    grid = [[0, 1], [1, 1]]
    a_star = AStarGrid(grid, (0, 0), (1, 1))
    with pytest.raises(ValueError, match="Start position.*is a wall")

def test_wall_end():
    grid = [[1, 1], [1, 0]]
    a_star = AStarGrid(grid, (0, 0), (1, 1))
    with pytest.raises(ValueError, match="End position.*is a wall")

def test_out_of_bounds():
    grid = [[1, 1], [1, 1]]
    a_star = AStarGrid(grid, (0, 0), (2, 2))
    with pytest.raises(ValueError, match="End position.*is out of bounds")

def test_no_path():
    grid = [[1, 0], [0, 1]]
    a_star = AStarGrid(grid, (0, 0), (1, 1))
    assert a_star.find_path() is None

def test_weighted_path():
    grid = [
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ]
    a_star = AStarGrid(grid, (0, 0), (2, 2))
    path = a_star.find_path()
    assert len(path) == 5  # Should take the direct path avoiding the 5 cell
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Check that the path avoids the high-weight cell (1,1)
    assert (1, 1) not in path

grid = [
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
]
a_star = AStarGrid(grid, (0, 0), (2, 2))
path = a_star.find_path()  # Returns [(0,0), (0,1), (0,2), (1,2), (2,2)]