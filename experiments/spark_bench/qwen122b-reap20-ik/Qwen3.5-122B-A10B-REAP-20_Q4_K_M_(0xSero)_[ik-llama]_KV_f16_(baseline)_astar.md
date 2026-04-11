# Qwen3.5-122B-A10B-REAP-20 Q4_K_M (0xSero) [ik-llama] — KV f16 (baseline) — A* Pathfinding

## Output

Here is the complete implementation of the A* pathfinding algorithm on a weighted 2D grid, followed by the requested `pytest` test suite.

### Implementation: `a_star_grid.py`

```python
import heapq
from typing import List, Tuple, Optional
from math import inf

class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.
    
    Attributes:
        grid (List[List[int]]): The 2D grid where 0 represents a wall and 
                                positive integers represent movement costs.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with the provided grid.
        
        Args:
            grid: A 2D list of integers representing movement costs.
                  0 = impassable wall, >0 = cost to enter.
        """
        if not grid or not grid[0]:
            self.grid = []
            self.rows = 0
            self.cols = 0
            return

        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

        # Validate grid consistency (optional but good practice)
        for row in self.grid:
            if len(row) != self.cols:
                raise ValueError("Grid rows must have consistent column counts.")

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate the Manhattan distance heuristic between two points.
        
        Args:
            a: Start coordinate (row, col).
            b: End coordinate (row, col).
            
        Returns:
            The Manhattan distance (|r1-r2| + |c1-c2|).
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[Tuple[int, int], int]]:
        """
        Get valid neighbors for a given node in 4 directions (up, down, left, right).
        
        Args:
            node: Current coordinate (row, col).
            
        Returns:
            A list of tuples containing (neighbor_coords, movement_cost).
            Movement cost is the value of the neighbor cell in the grid.
        """
        r, c = node
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                cost = self.grid[nr][nc]
                if cost > 0: # 0 is a wall
                    neighbors.append(((nr, nc), cost))
        
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using the A* algorithm.
        
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

        # Edge case: Start is a wall or End is a wall
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None

        # Edge case: Start equals End
        if start == end:
            return [start]

        # Priority queue: (f_score, counter, node)
        # Counter is used to break ties in heap to ensure stability
        open_set = []
        counter = 0
        
        # g_score: cost from start to current node
        g_score = {start: 0}
        
        # To reconstruct path
        came_from = {}

        # Initial f_score = g_score + heuristic
        h_start = self._heuristic(start, end)
        heapq.heappush(open_set, (h_start, counter, start))
        counter += 1

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                curr = current
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.append(start)
                return path[::-1]

            for neighbor, move_cost in self._get_neighbors(current):
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1

        return None
```

### Test Suite: `test_a_star_grid.py`

This file contains 6 comprehensive tests using `pytest`. It verifies path validity, optimality, and edge case handling.

```python
import pytest
from typing import List, Tuple
from a_star_grid import AStarGrid

def calculate_path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Helper to calculate the total cost of a path (sum of costs of entered cells)."""
    if not path:
        return 0
    # The cost of the path is the sum of the costs of the cells we step INTO.
    # Usually, the start cell cost is not included in the movement cost, 
    # but the problem says "cost to enter that cell". 
    # Standard interpretation: Sum of grid[r][c] for all cells in path EXCEPT start.
    # However, if the problem implies the cost of the path is the sum of all nodes visited including start,
    # we adjust. Based on standard A* on weighted grids, we sum the costs of the edges/nodes traversed.
    # Let's assume the cost is the sum of the values of the cells we move into.
    # If start == end, cost is 0 (no movement).
    
    total = 0
    for i, (r, c) in enumerate(path):
        if i == 0:
            continue # Don't count start cell cost as we are already there
        total += grid[r][c]
    return total

def is_path_valid(grid: List[List[int]], path: List[Tuple[int, int]], start: Tuple[int, int], end: Tuple[int, int]) -> bool:
    """Checks if the path is contiguous, within bounds, and avoids walls."""
    if not path:
        return False
    if path[0] != start or path[-1] != end:
        return False
    
    for i, (r, c) in enumerate(path):
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
            return False
        if grid[r][c] == 0:
            return False
        
        if i > 0:
            prev_r, prev_c = path[i-1]
            dist = abs(r - prev_r) + abs(c - prev_c)
            if dist != 1:
                return False
    return True

class TestAStarGrid:

    def test_simple_path_uniform_grid(self):
        """Test 1: Simple path on a uniform grid (all costs = 1)."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is not None
        assert is_path_valid(grid, path, (0, 0), (2, 2))
        # Optimal cost for 3x3 uniform grid (Manhattan distance 4 steps)
        # Path length is 5 nodes. Steps = 4. Cost = 4 * 1 = 4.
        cost = calculate_path_cost(grid, path)
        assert cost == 4

    def test_path_around_obstacles(self):
        """Test 2: Path must go around obstacles (0s)."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 3))
        
        assert path is not None
        assert is_path_valid(grid, path, (0, 0), (2, 3))
        # Must go down to row 2 or up (if possible, but here only down)
        # Path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(2,3)
        # Cost: 1+1+1+1+1 = 5
        cost = calculate_path_cost(grid, path)
        assert cost == 5

    def test_weighted_grid_prefers_lower_cost(self):
        """Test 3: Weighted grid where path prefers lower-cost cells over shorter distance."""
        # Grid:
        # S 1 1
        # 1 10 1
        # 1 1 1 E
        # Direct path via middle (1,1) costs 10.
        # Path around (down then right) costs 1+1+1+1 = 4.
        grid = [
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is not None
        assert is_path_valid(grid, path, (0, 0), (2, 2))
        
        cost = calculate_path_cost(grid, path)
        # The optimal path should avoid the 10.
        # Path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2) OR (0,0)->(0,1)->(0,2)->(1,2)->(2,2)
        # Both have cost 4.
        assert cost == 4
        # Ensure we didn't take the expensive route
        assert (1, 1) not in path

    def test_no_path_exists_fully_blocked(self):
        """Test 4: No path exists because the destination is surrounded by walls."""
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 1]
        ]
        astar = AStarGrid(grid)
        # Start at (0,0), End at (2,2). (2,2) is reachable? 
        # Let's make (2,2) unreachable.
        grid_blocked = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 0] # (2,2) is 0 (wall)
        ]
        astar_blocked = AStarGrid(grid_blocked)
        path = astar_blocked.find_path((0, 0), (2, 2))
        assert path is None

        # Another case: End is reachable but surrounded by walls from start
        grid_isolated = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ]
        astar_iso = AStarGrid(grid_isolated)
        path_iso = astar_iso.find_path((0, 0), (2, 2))
        assert path_iso is None

    def test_start_equals_end(self):
        """Test 5: Start and end are the same coordinate."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((1, 1), (1, 1))
        
        assert path is not None
        assert path == [(1, 1)]
        assert calculate_path_cost(grid, path) == 0

    def test_invalid_coordinates(self):
        """Test 6: Start or end coordinates are out of bounds."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Test start out of bounds
        with pytest.raises(ValueError, match="out of bounds"):
            astar.find_path((-1, 0), (0, 0))
            
        with pytest.raises(ValueError, match="out of bounds"):
            astar.find_path((0, 0), (2, 0))
            
        with pytest.raises(ValueError, match="out of bounds"):
            astar.find_path((0, 0), (0, 5))

    def test_start_or_end_is_wall(self):
        """Additional edge case: Start or End is a wall (0)."""
        grid = [
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0]
        ]
        astar = AStarGrid(grid)
        
        # Start is wall
        path_start_wall = astar.find_path((0, 0), (1, 1))
        assert path_start_wall is None
        
        # End is wall
        path_end_wall = astar.find_path((1, 1), (2, 2))
        assert path_end_wall is None
```

### How to Run

1.  Save the implementation code as `a_star_grid.py`.
2.  Save the test code as `test_a_star_grid.py`.
3.  Ensure you have `pytest` installed:
    ```bash
    pip install pytest
    ```
4.  Run the tests:
    ```bash
    pytest test_a_star_grid.py -v
    ```

### Key Design Decisions

1.  **Cost Definition**: The problem states "positive int = cost to enter that cell". Therefore, the total path cost is the sum of the grid values of all cells in the path **excluding** the start cell (since we are already there).
2.  **Heuristic**: Manhattan distance is used as required. Since movement is restricted to 4 directions, this is an admissible heuristic (never overestimates the true cost) provided the minimum movement cost is 1. If costs can be fractional or less than 1, the heuristic might need scaling, but for integer costs $\ge 1$, it is safe.
3.  **Tie Breaking**: A `counter` is added to the heap tuple `(f_score, counter, node)`. This ensures that if two nodes have the same `f_score`, the one pushed earlier is processed first, preventing potential infinite loops in specific edge cases and ensuring deterministic behavior.
4.  **Error Handling**: `ValueError` is raised strictly for out-of-bounds coordinates. If a coordinate is valid but points to a wall (0), the function returns `None` as no path can exist starting or ending on a wall.
