# Qwen 3.5 9B Q4_K_M — astar

## Output

```python
from typing import List, Tuple, Optional
import heapq
import math

class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.
    
    Supports 4-directional movement with Manhattan heuristic and optimal cost pathfinding.
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid with a 2D grid of movement costs.
        
        Args:
            grid: 2D list where each cell contains:
                  - 0: impassable wall
                  - positive int: cost to enter that cell
                  
        Raises:
            ValueError: If grid is empty or contains invalid values
        """
        if not grid or not all(isinstance(row, list) for row in grid):
            raise ValueError("Grid must be a non-empty 2D list")
        
        if not all(len(row) == len(grid[0]) for row in grid):
            raise ValueError("All rows must have the same length")
        
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        for row in grid:
            for cost in row:
                if cost < 0:
                    raise ValueError("Movement costs must be non-negative")
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting position (row, col)
            end: Ending position (row, col)
            
        Returns:
            List of (row, col) tuples representing the path, or None if no path exists.
            Path includes start and end positions.
            
        Raises:
            ValueError: If start or end positions are out of bounds or are walls
        """
        start_row, start_col = start
        end_row, end_col = end
        
        # Validate start and end positions
        if not self._is_valid_position(start_row, start_col):
            raise ValueError(f"Start position {start} is out of bounds or is a wall")
        
        if not self._is_valid_position(end_row, end_col):
            raise ValueError(f"End position {end} is out of bounds or is a wall")
        
        # Handle start == end case
        if start == end:
            return [start]
        
        # A* algorithm
        open_set = []
        heapq.heappush(open_set, (self._f_score(start, end), start))
        
        came_from = {}
        g_score = {start: 0}
        closed_set = set()
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            
            if current == end:
                # Reconstruct path
                path = []
                current = end
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            closed_set.add(current)
            
            # Explore neighbors (4-directional)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not self._is_valid_position(neighbor[0], neighbor[1]):
                    continue
                
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                movement_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + movement_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._h_score(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))
        
        return None
    
    def _is_valid_position(self, row: int, col: int) -> bool:
        """Check if a position is within bounds and not a wall."""
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        return self.grid[row][col] != 0
    
    def _h_score(self, current: Tuple[int, int], target: Tuple[int, int]) -> float:
        """Calculate Manhattan heuristic distance."""
        return abs(current[0] - target[0]) + abs(current[1] - target[1])
    
    def _f_score(self, current: Tuple[int, int], target: Tuple[int, int]) -> float:
        """Calculate f_score = g_score + h_score."""
        return self._h_score(current, target)
```

```python
import pytest
from typing import List, Tuple, Optional

class TestAStarGrid:
    """Test cases for AStarGrid class."""
    
    def test_simple_path(self):
        """Test basic pathfinding on a simple grid."""
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is not None
        assert path == [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2)]
    
    def test_start_equals_end(self):
        """Test when start and end positions are the same."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (0, 0))
        
        assert path == [(0, 0)]
    
    def test_no_path_exists(self):
        """Test when no path exists due to walls."""
        grid = [
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is None
    
    def test_out_of_bounds_start(self):
        """Test ValueError when start position is out of bounds."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        
        with pytest.raises(ValueError):
            astar.find_path((3, 3), (0, 0))
    
    def test_out_of_bounds_end(self):
        """Test ValueError when end position is out of bounds."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (3, 3))
    
    def test_wall_at_start(self):
        """Test ValueError when start position is a wall."""
        grid = [
            [0, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (1, 1))
    
    def test_optimal_path_cost(self):
        """Test that the path found is optimal (minimum total cost)."""
        grid = [
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is not None
        # Path should avoid the high-cost cell (1, 1)
        assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
        # Calculate total cost
        total_cost = sum(grid[r][c] for r, c in path[1:])  # Exclude start
        assert total_cost == 5  # 1+1+1+1+1
```
