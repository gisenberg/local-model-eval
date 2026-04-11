# Qwen 3.5 27B Opus-Distilled Q4_K_M (f16/f16 @ 65K baseline) — astar

**Hypothesis:** H2 baseline: does f16 KV at 65K fit at all? 9.6 GB KV + 16.5 weights + 8 compute = 34 GB likely OOM

## Output

Here is the complete implementation of the A* pathfinding algorithm in Python, followed by the requested pytest test suite.

### Implementation

```python
import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* Grid.
        
        Args:
            grid: A 2D list where values represent movement cost.
                  0 represents an impassable wall.
                  Positive integers represent the cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        
        # Validate grid consistency
        if not all(len(row) == self.cols for row in self.grid):
            raise ValueError("Grid rows must have equal length.")

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

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

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
        # 1. Input Validation
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start position {start} is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End position {end} is out of bounds.")
            
        # 2. Edge Case: Start equals End
        if start == end:
            return [start]

        # 3. Edge Case: Start or End is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # 4. A* Initialization
        # Priority Queue: (f_score, g_score, current_pos)
        # We use g_score as a tie-breaker in the tuple to ensure stability, 
        # though strictly f_score is the primary sort key.
        open_set = [(self._heuristic(start, end), 0, start)]
        
        # Track visited nodes to avoid cycles and redundant processing
        visited = set()
        
        # Track parent pointers to reconstruct the path
        # parent[child] = parent_node
        parent = {}
        
        # Track the lowest g_score found so far for each node
        g_scores = {start: 0}

        while open_set:
            # Pop the node with the lowest f_score
            _, current_g, current = heapq.heappop(open_set)
            
            # If we reached the end
            if current == end:
                return self._reconstruct_path(parent, start, end)
            
            # If we've already processed this node with a lower cost, skip
            if current in visited:
                continue
            
            visited.add(current)
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                if neighbor in visited:
                    continue
                
                # Cost to move to neighbor
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = current_g + move_cost
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    
                    # Push to open set
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    
                    # Record parent
                    parent[neighbor] = current

        # No path found
        return None

    def _reconstruct_path(self, parent: dict, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from end to start using parent pointers."""
        path = [end]
        current = end
        while current != start:
            if current not in parent:
                return None # Should not happen if logic is correct
            current = parent[current]
            path.append(current)
        return path[::-1]
```

### Pytest Tests

Save the following code in a file named `test_astar.py` to run the tests.

```python
import pytest
from typing import List, Tuple, Optional
import sys
import os

# Ensure the implementation file is importable
# Assuming the implementation is in a file named 'astar_impl.py' or similar.
# For this example, we will import the class directly if run in the same environment.
# If running separately, ensure the path is correct.
try:
    from astar_impl import AStarGrid
except ImportError:
    # Fallback for single-file testing if needed, though usually separate files are preferred.
    # Here we assume the user has the class defined above in a module named 'astar_impl'
    # or we define it here for the sake of the test block if running inline.
    # To make this runnable immediately, we define the class here again briefly or assume import.
    # For the purpose of this output, I will assume the user saves the class above as 'astar_impl.py'.
    pass

# --- Test Data ---

def test_simple_path():
    """Test a simple straight path."""
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

def test_weighted_optimal_path():
    """Test that the algorithm chooses the lower cost path over the shorter distance."""
    # Grid:
    # 1 1 1 1
    # 1 9 9 1
    # 1 1 1 1
    # Path 1 (Top): 1->1->1->1 (Cost 3)
    # Path 2 (Bottom): 1->1->1->1 (Cost 3)
    # Let's make the middle expensive to force a detour if direct is blocked, 
    # or compare two routes with different weights.
    
    grid = [
        [1, 1, 1, 1],
        [1, 9, 9, 1],
        [1, 1, 1, 1]
    ]
    # Start (0,0), End (2,3)
    # Route A: (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3) Cost: 1+1+1+1+1+1 = 6
    # Route B: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(2,3) Cost: 1+1+1+1+1+1 = 6
    # Let's modify to make one clearly cheaper.
    
    grid = [
        [1, 1, 1, 1],
        [1, 9, 9, 1],
        [1, 1, 1, 1]
    ]
    # Actually, let's do a classic weighted choice.
    # Start (0,0), End (0,2)
    # Direct: (0,0)->(0,1)->(0,2) Cost: 1+1 = 2
    # Detour: (0,0)->(1,0)->(1,1)->(1,2)->(0,2) Cost: 1+1+1+1 = 4
    # But what if (0,1) is expensive?
    
    grid = [
        [1, 10, 1],
        [1, 1, 1]
    ]
    # Start (0,0), End (0,2)
    # Direct: 1 -> 10 -> 1 = 11
    # Detour: 1 -> 1 -> 1 -> 1 -> 1 = 5
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is not None
    # The path should go down to row 1 to avoid the 10
    assert (1, 0) in path
    assert (0, 1) not in path

def test_wall_blocking():
    """Test that walls block the path."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Start (0,0), End (0,2) - blocked by wall at (0,1) and (1,1)
    # Wait, (1,1) is wall, (0,1) is wall.
    # Path must go (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2)
    # Let's make it impossible.
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_start_equals_end():
    """Test when start and end are the same."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_out_of_bounds_error():
    """Test ValueError for out of bounds coordinates."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
        
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_start_or_end_is_wall():
    """Test that starting or ending on a wall returns None."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Start on wall
    assert astar.find_path((0, 0), (1, 1)) is None
    
    # End on wall
    grid2 = [
        [1, 1],
        [1, 0]
    ]
    astar2 = AStarGrid(grid2)
    assert astar2.find_path((0, 0), (1, 1)) is None
```

### How to Run

1.  Save the first code block into a file named `astar_impl.py`.
2.  Save the second code block into a file named `test_astar.py`.
3.  Run the tests using pytest:
    ```bash
    pytest test_astar.py -v
    ```

### Explanation of Key Logic

1.  **Heuristic**: We use the Manhattan distance (`abs(x1-x2) + abs(y1-y2)`). This is admissible (never overestimates) for 4-directional movement on a grid, ensuring optimality.
2.  **Priority Queue**: We use `heapq` to store tuples of `(f_score, g_score, position)`.
    *   `g_score`: The actual cost from start to current node.
    *   `h_score`: The estimated cost from current node to end.
    *   `f_score`: `g_score + h_score`.
3.  **Visited Set**: We maintain a `visited` set to ensure we don't process the same node multiple times with a worse or equal cost, which prevents infinite loops and improves efficiency.
4.  **Reconstruction**: We store a `parent` dictionary mapping each node to its predecessor. Once the end node is reached, we backtrack from end to start and reverse the list.
5.  **Cost Calculation**: The cost to move to a neighbor is defined by the value in the grid at that neighbor's position (`self.grid[neighbor[0]][neighbor[1]]`).
