import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    The grid is represented as a list of lists of numbers where:
    - 0 represents a wall (impassable)
    - Positive numbers represent the cost to enter the cell
    """
    def __init__(self, grid: List[List[int]]) -> None:
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # Ensure rectangular grid
        if any(len(row) != self.cols for row in self.grid):
            raise ValueError("Grid must be rectangular (all rows must have equal length).")

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using the A* algorithm.
        
        Args:
            start: (row, col) tuple representing the starting position.
            end: (row, col) tuple representing the target position.
            
        Returns:
            A list of (row, col) tuples representing the optimal path,
            including start and end. Returns None if no valid path exists.
            
        Raises:
            ValueError: If start or end is out of bounds or positioned on a wall.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds.")
            
        # Validate walls at start/end
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall.")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall.")
            
        # Handle start == end
        if start == end:
            return [start]
            
        # Priority queue: (f_score, tie_breaker, position)
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))
        
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        tie_breaker = 1
        
        # 4-directional movement: right, left, down, up
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
                
            for dr, dc in directions:
                nr, nc = current[0] + dr, current[1] + dc
                
                # Skip out-of-bounds and walls
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr][nc] == 0:
                    continue
                    
                weight = self.grid[nr][nc]
                tentative_g = g_score[current] + weight
                
                # Found a better path to neighbor
                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    # Manhattan heuristic
                    f_score = tentative_g + abs(nr - end[0]) + abs(nc - end[1])
                    heapq.heappush(open_set, (f_score, tie_breaker, (nr, nc)))
                    tie_breaker += 1
                    
        return None

import pytest

def test_basic_pathfinding():
    """Test standard pathfinding on an open grid."""
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
    # Shortest path in 3x3 grid is 5 nodes (4 steps)
    assert len(path) == 5

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_no_path_blocked_by_walls():
    """Test that None is returned when the target is completely enclosed by walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError, match="Start position is out of bounds"):
        astar.find_path((-1, 0), (1, 1))
        
    with pytest.raises(ValueError, match="End position is out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_at_start_or_end_raises_value_error():
    """Test that placing start or end on a wall raises ValueError."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError, match="Start position is a wall"):
        astar.find_path((0, 0), (1, 1))
        
    grid2 = [[1, 0], [1, 1]]
    astar2 = AStarGrid(grid2)
    with pytest.raises(ValueError, match="End position is a wall"):
        astar2.find_path((0, 0), (0, 1))

def test_weighted_optimality():
    """Test that A* chooses the lower-cost path over a shorter/higher-cost path."""
    # Center cell has high weight (10), surrounding cells have weight (1)
    grid = [
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    # Optimal path must avoid the expensive center cell
    assert (1, 1) not in path
    # Should take a path of length 5 (cost = 4) instead of going through center (cost = 12)
    assert len(path) == 5