from __future__ import annotations
import heapq
from typing import List, Tuple, Optional, Set, Dict

class AStarGrid:
    """
    A* pathfinding implementation for a weighted 2D grid.
    
    Args:
        grid: 2D list of integers where 0 represents walls and positive integers represent walkable cells with their weights.
        start: Tuple of (row, col) coordinates for the starting position.
        end: Tuple of (row, col) coordinates for the ending position.
    """
    
    def __init__(self, grid: List[List[int]], start: Tuple[int, int], end: Tuple[int, int]) -> None:
        self.grid = grid
        self.start = start
        self.end = end
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        
        # Validate start and end positions
        if not self._is_valid_position(start):
            raise ValueError(f"Start position {start} is out of bounds or is a wall")
        if not self._is_valid_position(end):
            raise ValueError(f"End position {end} is out of bounds or is a wall")
            
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall")

    def find_path(self) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Returns:
            List of (row, col) tuples representing the path from start to end, 
            or None if no path exists.
        """
        if self.start == self.end:
            return [self.start]
            
        open_set = []
        heapq.heappush(open_set, self._make_node(self.start, 0))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {self.start: 0}
        closed_set: Set[Tuple[int, int]] = set()
        
        while open_set:
            current = heapq.heappop(open_set)
            current_pos = current[2]
            
            if current_pos == self.end:
                return self._reconstruct_path(came_from, current_pos)
                
            closed_set.add(current_pos)
            
            for neighbor in self._get_neighbors(current_pos):
                if neighbor in closed_set:
                    continue
                    
                tentative_g_score = current[0] + self._get_weight(neighbor)
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_pos
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self._heuristic(neighbor)
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))
                    
        return None

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds and not a wall."""
        rows, cols = self.rows, self.cols
        row, col = pos
        return 0 <= row < rows and 0 <= col < cols and self.grid[row][col] != 0

    def _get_weight(self, pos: Tuple[int, int]) -> int:
        """Get the weight of a cell (0 for walls, positive for walkable)."""
        return self.grid[pos[0]][pos[1]]

    def _heuristic(self, pos: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(pos[0] - self.end[0]) + abs(pos[1] - self.end[1])

    def _make_node(self, pos: Tuple[int, int], g_score: float) -> Tuple[float, float, Tuple[int, int]]:
        """Create a node for the priority queue."""
        return (g_score + self._heuristic(pos), g_score, pos)

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get 4-directional neighbors (up, down, left, right)."""
        rows, cols = self.rows, self.cols
        row, col = pos
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < rows and 0 <= new_col < cols and self.grid[new_row][new_col] != 0:
                neighbors.append((new_row, new_col))
                
        return neighbors

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to current position."""
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
    finder = AStarGrid(grid, (0, 0), (0, 0))
    assert finder.find_path() == [(0, 0)]

def test_no_path_exists():
    grid = [[1, 0], [0, 1]]
    finder = AStarGrid(grid, (0, 0), (1, 1))
    assert finder.find_path() is None

def test_wall_at_start():
    grid = [[0, 1], [1, 1]]
    with pytest.raises(ValueError, match="Start position is a wall"):
        AStarGrid(grid, (0, 0), (1, 1))

def test_wall_at_end():
    grid = [[1, 1], [1, 0]]
    with pytest.raises(ValueError, match="End position is a wall"):
        AStarGrid(grid, (0, 0), (1, 1))

def test_out_of_bounds_start():
    grid = [[1, 1], [1, 1]]
    with pytest.raises(ValueError, match="Start position"):
        AStarGrid(grid, (2, 0), (1, 1))

def test_out_of_bounds_end():
    grid = [[1, 1], [1, 1]]
    with pytest.raises(ValueError, match="End position"):
        AStarGrid(grid, (0, 0), (0, 2))

def test_weighted_path():
    grid = [
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ]
    finder = AStarGrid(grid, (0, 0), (2, 2))
    path = finder.find_path()
    # Path should avoid the center cell (weight 5)
    assert path[1] == (0, 1)  # First step should be right, not down
    assert len(path) == 5  # 5 steps for 3x3 grid