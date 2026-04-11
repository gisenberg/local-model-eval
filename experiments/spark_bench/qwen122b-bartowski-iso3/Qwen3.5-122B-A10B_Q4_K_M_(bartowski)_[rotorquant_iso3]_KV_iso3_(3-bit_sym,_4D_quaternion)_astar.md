# Qwen3.5-122B-A10B Q4_K_M (bartowski) [rotorquant iso3] — KV iso3 (3-bit sym, 4D quaternion) — A* Pathfinding

## Output

Here is the complete implementation of the A* pathfinding algorithm in Python, followed by the required `pytest` test suite.

### Implementation (`a_star.py`)

```python
import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.
    
    Attributes:
        grid: 2D list where 0 represents an impassable wall and positive integers
              represent the movement cost to enter that cell.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with the provided cost grid.
        
        Args:
            grid: A 2D list of integers representing movement costs.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate the Manhattan distance heuristic between two points.
        
        Args:
            a: Current coordinate (row, col).
            b: Target coordinate (row, col).
            
        Returns:
            Manhattan distance (|x1-x2| + |y1-y2|).
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid 4-directional neighbors for a given node.
        
        Args:
            node: Current coordinate (row, col).
            
        Returns:
            List of valid neighbor coordinates.
        """
        r, c = node
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] != 0: # Check if not a wall
                    neighbors.append((nr, nc))
        
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting coordinate (row, col).
            end: Ending coordinate (row, col).
            
        Returns:
            A list of coordinates representing the path from start to end (inclusive),
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start coordinate {start} is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End coordinate {end} is out of bounds.")

        # Handle edge case: start is a wall or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Handle edge case: start equals end
        if start == end:
            return [start]

        # Priority queue: (f_score, counter, node)
        # Counter is used to break ties in heap to ensure stability
        counter = 0
        open_set = []
        heapq.heappush(open_set, (0, counter, start))
        
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score: cost from start to current node
        g_score: dict[Tuple[int, int], int] = {start: 0}
        
        # f_score: g_score + heuristic
        f_score: dict[Tuple[int, int], int] = {start: self._heuristic(start, end)}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._get_neighbors(current):
                # Cost to enter the neighbor cell
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    f_score[neighbor] = f
                    
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))

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
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1] # Reverse to get start -> end
```

### Test Suite (`test_a_star.py`)

```python
import pytest
from a_star import AStarGrid

# Helper to calculate total path cost
def calculate_path_cost(grid, path):
    if not path:
        return 0
    total = 0
    for r, c in path:
        total += grid[r][c]
    return total

def test_simple_path_uniform_grid():
    """
    Test 1: Simple path on a uniform grid.
    Grid is 3x3 with all costs 1. Path should be straight or L-shaped with equal cost.
    """
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
    # Manhattan distance is 4 steps (2 down + 2 right), cost = 1 * 5 (including start)
    # Actually, cost is sum of cells entered. 
    # Path length 5 cells. Cost = 5.
    assert len(path) == 5
    assert calculate_path_cost(grid, path) == 5

def test_path_around_obstacles():
    """
    Test 2: Path around obstacles.
    A wall blocks the direct path, forcing a detour.
    """
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 3))
    
    assert path is not None
    # Verify no wall is in the path
    for r, c in path:
        assert grid[r][c] != 0
    
    # Verify optimality: Must go around the wall.
    # Path: (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3) OR similar
    # Cost calculation: 1+1+1+1+1+1 = 6
    assert calculate_path_cost(grid, path) == 6

def test_weighted_grid_optimality():
    """
    Test 3: Weighted grid where path prefers lower-cost cells.
    A longer path with lower costs should be chosen over a shorter path with high costs.
    """
    grid = [
        [1, 1, 1, 1],
        [1, 10, 10, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Start (0,0) to End (2,3)
    # Option A (Direct through middle): (0,0)->(1,0)->(1,1)->(1,2)->(1,3)->(2,3)
    # Cost: 1 + 1 + 10 + 10 + 1 + 1 = 23
    # Option B (Around top/bottom): (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3)
    # Cost: 1 + 1 + 1 + 1 + 1 + 1 = 6
    
    path = astar.find_path((0, 0), (2, 3))
    assert path is not None
    
    total_cost = calculate_path_cost(grid, path)
    assert total_cost == 6, f"Expected cost 6, got {total_cost}. Path did not avoid high cost cells."
    
    # Verify the path does not touch the high cost cells (10)
    for r, c in path:
        assert grid[r][c] != 10

def test_no_path_exists():
    """
    Test 4: No path exists (fully blocked).
    Start and end are separated by a wall of 0s.
    """
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is None

def test_start_equals_end():
    """
    Test 5: Start equals end.
    Should return a list containing only the start coordinate.
    """
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    
    assert path == [(1, 1)]
    assert calculate_path_cost(grid, path) == 1

def test_invalid_coordinates():
    """
    Test 6: Invalid coordinates (out of bounds).
    Should raise ValueError.
    """
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
    
    # Test start out of bounds (col)
    with pytest.raises(ValueError):
        astar.find_path((0, 10), (0, 0))

def test_start_or_end_is_wall():
    """
    Test 7: Start or end is a wall.
    Should return None.
    """
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Start is wall
    assert astar.find_path((0, 0), (1, 1)) is None
    
    # End is wall
    assert astar.find_path((1, 1), (0, 0)) is None
```

### How to Run

1.  Save the first code block as `a_star.py`.
2.  Save the second code block as `test_a_star.py`.
3.  Ensure `pytest` is installed (`pip install pytest`).
4.  Run the tests from the terminal:
    ```bash
    pytest test_a_star.py -v
    ```

### Key Implementation Details

*   **Heuristic**: Uses Manhattan distance (`|x1-x2| + |y1-y2|`), which is admissible for 4-directional movement.
*   **Optimality**: The algorithm uses a `g_score` (actual cost from start) and `f_score` (g + heuristic). It only updates a node if a strictly lower cost path is found, guaranteeing the optimal path for positive weights.
*   **Heap Usage**: `heapq` is used with a tuple `(f_score, counter, node)`. The `counter` is a tie-breaker to ensure the heap remains stable when multiple nodes have the same `f_score`.
*   **Edge Cases**:
    *   **Start == End**: Returns immediately with `[start]`.
    *   **Walls**: Checks `grid[r][c] == 0` before adding to neighbors and explicitly checks start/end validity.
    *   **Bounds**: Raises `ValueError` immediately if coordinates are invalid.
*   **Cost Calculation**: The cost of a path includes the cost of the starting cell and the ending cell, as per the requirement "cost to enter that cell" (and logically, you must "enter" the start to begin).
