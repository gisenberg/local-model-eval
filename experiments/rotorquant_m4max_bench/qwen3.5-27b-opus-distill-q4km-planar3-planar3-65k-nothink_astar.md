# Qwen 3.5 27B Opus-Distilled Q4_K_M (planar3/planar3 @ 65K) — astar

**Hypothesis:** H2 probe: can we reach 64K context on Qwen 27B with rotorquant where f16 OOMs?

## Output

Here is the complete implementation of the A* pathfinding algorithm in Python, followed by the requested pytest test suite.

### Implementation

```python
import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D list of integers.
        
        Args:
            grid: A 2D list where 0 represents a wall (impassable) 
                   and positive integers represent the cost to enter a cell.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
            
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # Validate grid consistency
        for row in grid:
            if len(row) != self.cols:
                raise ValueError("Grid rows must have equal length.")
            for val in row:
                if val < 0:
                    raise ValueError("Grid values must be non-negative.")

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within bounds and not a wall."""
        r, c = pos
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r][c] > 0
        return False

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid 4-directional neighbors."""
        r, c = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self._is_valid((nr, nc)):
                neighbors.append((nr, nc))
        return neighbors

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Tuple (row, col) representing the starting position.
            end: Tuple (row, col) representing the target position.
            
        Returns:
            A list of tuples representing the path from start to end, 
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Input validation
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds.")
            
        # If start is a wall, no path possible
        if self.grid[start[0]][start[1]] == 0:
            return None
            
        # Optimization: Start equals End
        if start == end:
            return [start]

        # Priority Queue: (f_score, g_score, current_pos)
        # We use a counter to break ties deterministically if needed, 
        # though strictly not required for correctness here.
        open_set = [(self._heuristic(start, end), 0, start)]
        
        # Track g_scores (cost from start)
        g_scores = {start: 0}
        
        # Track parents for path reconstruction
        came_from = {}
        
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            
            # If we reached the goal
            if current == end:
                return self._reconstruct_path(came_from, current)
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                # Cost to move to neighbor
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = current_g + move_cost
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    came_from[neighbor] = current
                    
        return None

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from the came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]
```

### Pytest Tests

Save this code in a file named `test_astar.py` to run the tests.

```python
import pytest
from typing import List, Tuple, Optional
import sys

# Import the class from the implementation file
# Assuming the implementation is in a file named 'astar_impl.py' or similar.
# For this standalone example, we will import it dynamically or assume it's in the same scope.
# In a real scenario, you would do: from astar_impl import AStarGrid
# Here we define it inline for the test runner to work immediately if copied together.

import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        for row in grid:
            if len(row) != self.cols:
                raise ValueError("Grid rows must have equal length.")
            for val in row:
                if val < 0:
                    raise ValueError("Grid values must be non-negative.")

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r][c] > 0
        return False

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self._is_valid((nr, nc)):
                neighbors.append((nr, nc))
        return neighbors

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds.")
            
        if self.grid[start[0]][start[1]] == 0:
            return None
            
        if start == end:
            return [start]

        open_set = [(self._heuristic(start, end), 0, start)]
        g_scores = {start: 0}
        came_from = {}
        
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            
            if current == end:
                return self._reconstruct_path(came_from, current)
            
            for neighbor in self._get_neighbors(current):
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = current_g + move_cost
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    came_from[neighbor] = current
                    
        return None

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

# --- TESTS ---

def test_simple_path():
    """Test a basic path on a small grid."""
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
    # Length should be 5 (Manhattan distance + 1)
    assert len(path) == 5

def test_wall_blocking():
    """Test that walls block the path."""
    grid = [
        [1, 1, 1],
        [0, 0, 0], # Wall row
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 0))
    assert path is None

def test_optimal_weighted_path():
    """Test that the algorithm chooses the lower cost path."""
    # Grid:
    # 1 1 1
    # 1 9 1
    # 1 1 1
    # Path around the 9 should be cheaper than going through it.
    grid = [
        [1, 1, 1],
        [1, 9, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    # Calculate cost of path
    total_cost = 0
    for r, c in path[1:]: # Skip start
        total_cost += grid[r][c]
        
    # Optimal path goes around: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) cost = 1+1+1+1 = 4
    # Or (0,0)->(1,0)->(2,0)->(2,1)->(2,2) cost = 1+1+1+1 = 4
    # Path through center: (0,0)->(0,1)->(1,1)->(2,1)->(2,2) cost = 1+9+1+1 = 12
    assert total_cost == 4

def test_start_equals_end():
    """Test when start and end are the same."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_out_of_bounds_start():
    """Test ValueError for out of bounds start."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((5, 5), (0, 0))

def test_out_of_bounds_end():
    """Test ValueError for out of bounds end."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))
```
