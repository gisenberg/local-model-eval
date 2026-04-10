# Qwen3.5-122B-A10B Q4_K_M (bartowski) [ik-llama] — KV f16 (baseline) — A* Pathfinding

## Output

Here is the complete implementation of the A* pathfinding algorithm on a weighted 2D grid, followed by the requested `pytest` test suite.

### Implementation: `a_star_grid.py`

```python
import heapq
from typing import List, Tuple, Optional, Set

class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.
    
    Attributes:
        grid: A 2D list where 0 represents an impassable wall and positive integers
              represent the movement cost to enter that cell.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid with the provided cost map.
        
        Args:
            grid: 2D list of integers representing movement costs.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # Validate grid consistency (optional but good practice)
        for row in grid:
            if len(row) != self.cols:
                raise ValueError("Grid rows must have consistent length.")

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, r: int, c: int) -> bool:
        """Check if a cell is a wall (cost 0)."""
        return self.grid[r][c] == 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two points.
        
        Args:
            a: Start coordinate (row, col).
            b: End coordinate (row, col).
            
        Returns:
            Manhattan distance.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Tuple (row, col) representing the starting position.
            end: Tuple (row, col) representing the target position.
            
        Returns:
            A list of (row, col) tuples representing the path from start to end inclusive,
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate bounds
        if not self._is_valid(*start):
            raise ValueError(f"Start coordinates {start} are out of bounds.")
        if not self._is_valid(*end):
            raise ValueError(f"End coordinates {end} are out of bounds.")

        # Edge case: Start or End is a wall
        if self._is_wall(*start) or self._is_wall(*end):
            return None

        # Edge case: Start equals End
        if start == end:
            return [start]

        # Directions: Up, Down, Left, Right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Priority Queue: (f_score, g_score, row, col)
        # We use a counter to break ties if f_scores are equal, ensuring stability
        # though not strictly required for correctness.
        open_set = []
        heapq.heappush(open_set, (0, 0, start[0], start[1]))

        # Track visited nodes and their best g_scores
        g_scores: dict[Tuple[int, int], int] = {start: 0}
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_set:
            _, current_g, r, c = heapq.heappop(open_set)
            current = (r, c)

            # If we reached the goal, reconstruct path
            if current == end:
                return self._reconstruct_path(came_from, current)

            # Skip if we found a better path to this node already
            if current_g > g_scores.get(current, float('inf')):
                continue

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)

                if not self._is_valid(nr, nc) or self._is_wall(nr, nc):
                    continue

                # Cost to enter the neighbor cell
                move_cost = self.grid[nr][nc]
                tentative_g = current_g + move_cost

                if tentative_g < g_scores.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, nr, nc))

        return None

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from end to start using the came_from map.
        
        Args:
            came_from: Dictionary mapping each node to its predecessor.
            current: The end node.
            
        Returns:
            List of coordinates from start to end.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def calculate_path_cost(self, path: List[Tuple[int, int]]) -> int:
        """
        Calculate the total cost of a given path.
        Note: The cost includes the entry cost of every cell in the path except the start.
        
        Args:
            path: List of (row, col) tuples.
            
        Returns:
            Total movement cost.
        """
        if not path:
            return 0
        
        total_cost = 0
        # Start node cost is usually not counted as "movement" into it, 
        # but based on the problem description "cost to enter that cell",
        # we sum the costs of all cells except the start.
        for r, c in path[1:]:
            total_cost += self.grid[r][c]
        return total_cost
```

### Test Suite: `test_a_star_grid.py`

```python
import pytest
from a_star_grid import AStarGrid

# Helper to calculate path cost for assertions
def get_path_cost(grid, path):
    if not path:
        return 0
    total = 0
    for r, c in path[1:]: # Exclude start node cost
        total += grid[r][c]
    return total

def test_simple_path_uniform_grid():
    """Test 1: Simple path on a uniform grid (all costs = 1)."""
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
    
    # Optimal cost for 4-directional movement on 3x3 uniform grid is 4 steps (cost 4)
    # Path length should be 5 nodes (start + 4 moves)
    assert len(path) == 5
    assert astar.calculate_path_cost(path) == 4

def test_path_around_obstacles():
    """Test 2: Path finding around obstacles (walls)."""
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Start (0,0) to End (2,3)
    # Must go around the wall in the middle row
    path = astar.find_path((0, 0), (2, 3))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 3)
    
    # Verify no wall cells are in the path
    for r, c in path:
        assert grid[r][c] != 0

def test_weighted_grid_optimality():
    """Test 3: Weighted grid where path prefers lower-cost cells over shorter distance."""
    # Grid:
    # S 1 1
    # 1 10 1
    # 1 1 1
    # Direct path via middle (cost 10) is shorter in steps but expensive.
    # Path going around (cost 1+1+1+1+1) is longer in steps but cheaper.
    grid = [
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    
    # Calculate cost
    cost = astar.calculate_path_cost(path)
    
    # Path via middle: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) cost: 1+1+1+1 = 4? 
    # Wait, let's trace:
    # Option A (Top-Right): (0,0)->(0,1)->(0,2)->(1,2)->(2,2). Costs: 1+1+1+1 = 4.
    # Option B (Bottom-Left): (0,0)->(1,0)->(2,0)->(2,1)->(2,2). Costs: 1+1+1+1 = 4.
    # Option C (Through middle): (0,0)->(1,0)->(1,1)->(1,2)->(2,2). Costs: 1+10+1+1 = 13.
    # Option D (Through middle 2): (0,0)->(0,1)->(1,1)->(2,1)->(2,2). Costs: 1+10+1+1 = 13.
    
    # The algorithm should pick cost 4.
    assert cost == 4
    
    # Verify the path does NOT go through (1,1) which has cost 10
    assert (1, 1) not in path

def test_no_path_exists():
    """Test 4: No path exists because the destination is fully blocked."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    # Start (0,0), End (2,2). The column 1 is blocked, and row 2 col 1 is blocked.
    # Actually, let's make it fully blocked.
    grid_blocked = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar_blocked = AStarGrid(grid_blocked)
    path = astar_blocked.find_path((0, 0), (0, 2))
    
    assert path is None

def test_start_equals_end():
    """Test 5: Start and end coordinates are the same."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    
    assert path is not None
    assert path == [(1, 1)]
    assert astar.calculate_path_cost(path) == 0

def test_invalid_coordinates():
    """Test 6: Start or end coordinates are out of bounds."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Test start out of bounds
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
        
    # Test end out of bounds
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))
        
    # Test start out of bounds (negative col)
    with pytest.raises(ValueError):
        astar.find_path((0, -1), (0, 0))

def test_start_or_end_is_wall():
    """Additional Test: Start or End is a wall (0 cost)."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Start is wall
    path = astar.find_path((0, 0), (1, 1))
    assert path is None
    
    # End is wall
    path = astar.find_path((1, 0), (0, 0))
    assert path is None
```

### How to Run

1.  Save the implementation code as `a_star_grid.py`.
2.  Save the test code as `test_a_star_grid.py`.
3.  Ensure `pytest` is installed (`pip install pytest`).
4.  Run the tests:
    ```bash
    pytest test_a_star_grid.py -v
    ```

### Key Implementation Details

*   **Heuristic**: Uses Manhattan distance (`|x1-x2| + |y1-y2|`), which is admissible for 4-directional movement on a grid.
*   **Optimality**: The algorithm uses a Min-Heap (`heapq`) to always expand the node with the lowest $f(n) = g(n) + h(n)$. Since the heuristic is admissible and consistent, the first time the goal is popped from the heap, it is guaranteed to be the optimal path.
*   **Cost Handling**: The cost of a path is the sum of the values of the cells entered. The start cell's cost is not included in the movement cost (as you are already there), but every subsequent cell adds its value to the total.
*   **Edge Cases**:
    *   **Start == End**: Returns immediately with a list containing just the start coordinate.
    *   **Walls**: If start or end is 0, returns `None`.
    *   **Bounds**: Raises `ValueError` if coordinates are outside the grid dimensions.
*   **Type Hints**: Full type hinting is used for arguments, return types, and internal variables.
