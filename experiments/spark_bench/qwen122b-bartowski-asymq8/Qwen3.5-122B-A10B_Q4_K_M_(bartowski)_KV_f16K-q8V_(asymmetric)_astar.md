# Qwen3.5-122B-A10B Q4_K_M (bartowski) — KV f16K/q8V (asymmetric) — A* Pathfinding

## Output

Here is the complete implementation of the A* pathfinding algorithm on a weighted 2D grid, followed by the requested `pytest` test suite.

### Implementation: `a_star_grid.py`

```python
import heapq
from typing import List, Tuple, Optional, Set

class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.
    
    Attributes:
        grid (List[List[int]]): 2D list representing the grid where 0 is a wall 
                                and positive integers are movement costs.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with the provided grid.
        
        Args:
            grid: A 2D list of integers representing movement costs.
                  0 represents an impassable wall.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # Validate that all rows have the same length
        for row in grid:
            if len(row) != self.cols:
                raise ValueError("Grid must be rectangular (all rows must have the same length).")

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid 4-directional neighbors for a given position.
        
        Args:
            pos: A tuple (row, col) representing the current position.
            
        Returns:
            A list of valid (row, col) tuples for neighbors.
        """
        r, c = pos
        neighbors = []
        # Directions: Up, Down, Left, Right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                # Check if the neighbor is not a wall (cost > 0)
                if self.grid[nr][nc] > 0:
                    neighbors.append((nr, nc))
        
        return neighbors

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate the Manhattan distance heuristic between two points.
        
        Args:
            a: Start position (row, col).
            b: End position (row, col).
            
        Returns:
            The Manhattan distance between a and b.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using the A* algorithm.
        
        Args:
            start: A tuple (row, col) representing the starting position.
            end: A tuple (row, col) representing the target position.
            
        Returns:
            A list of (row, col) tuples representing the path from start to end,
            inclusive. Returns None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start coordinates {start} are out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End coordinates {end} are out of bounds.")

        # Handle edge case: start is a wall or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Handle edge case: start equals end
        if start == end:
            return [start]

        # Priority queue: (f_score, counter, row, col)
        # Counter is used to break ties in heap to ensure stability
        counter = 0
        open_set = []
        heapq.heappush(open_set, (0, counter, start[0], start[1]))
        
        # Track visited nodes to avoid re-processing
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score: cost from start to current node
        g_score: dict[Tuple[int, int], int] = {start: 0}
        
        # f_score: g_score + heuristic
        f_score: dict[Tuple[int, int], int] = {start: self._heuristic(start, end)}
        
        open_set_hash: Set[Tuple[int, int]] = {start}

        while open_set:
            _, _, current_r, current_c = heapq.heappop(open_set)
            current = (current_r, current_c)
            
            # If we reached the goal, reconstruct the path
            if current == end:
                return self._reconstruct_path(came_from, current)
            
            open_set_hash.discard(current)

            for neighbor in self._get_neighbors(current):
                # Cost to move from current to neighbor is the cost of the neighbor cell
                # (as per requirement: "cost to enter that cell")
                tentative_g_score = g_score[current] + self.grid[neighbor[0]][neighbor[1]]
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f = tentative_g_score + self._heuristic(neighbor, end)
                    f_score[neighbor] = f
                    
                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f, counter, neighbor[0], neighbor[1]))
                        open_set_hash.add(neighbor)
        
        # No path found
        return None

    def _reconstruct_path(self, came_from: dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from start to end by backtracking through came_from.
        
        Args:
            came_from: Dictionary mapping each node to its predecessor.
            current: The end node.
            
        Returns:
            A list of coordinates representing the path from start to end.
        """
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path
```

### Test Suite: `test_a_star_grid.py`

```python
import pytest
from typing import List, Tuple
from a_star_grid import AStarGrid

# Helper to calculate total path cost
def calculate_path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    if not path:
        return 0
    total = 0
    for r, c in path:
        total += grid[r][c]
    return total

class TestAStarGrid:
    
    def test_simple_path_uniform_grid(self):
        """Test 1: Simple path on a uniform grid (all costs are 1)."""
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
        
        # In a uniform grid, Manhattan distance is the optimal cost (excluding start cost if we count entry)
        # Here we count entry cost. Path length is 5 steps (0,0 -> 0,1 -> 0,2 -> 1,2 -> 2,2)
        # Cost = 1+1+1+1+1 = 5.
        # Any path of length 5 is optimal.
        assert len(path) == 5
        assert calculate_path_cost(grid, path) == 5

    def test_path_around_obstacles(self):
        """Test 2: Path finding around obstacles (walls)."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        # Start (0,0) to End (2,3). Must go around the wall in the middle.
        path = astar.find_path((0, 0), (2, 3))
        
        assert path is not None
        # Verify no wall cells are in the path
        for r, c in path:
            assert grid[r][c] > 0
        
        # Verify connectivity
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            assert abs(r1 - r2) + abs(c1 - c2) == 1

    def test_weighted_grid_optimality(self):
        """Test 3: Weighted grid where path prefers lower-cost cells."""
        # Grid with a "high cost" barrier in the middle row
        # Path should go around the high cost cells if the alternative is cheaper
        grid = [
            [1, 1, 1, 1, 1],
            [1, 10, 10, 10, 1],
            [1, 1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        # Start (0,0) to End (2,4)
        # Option A (Straight through middle): 1 + 10 + 10 + 10 + 1 = 32 (approx, depends on exact path)
        # Option B (Go around top/bottom): 1+1+1+1+1+1+1+1+1 = 9 (approx)
        # The algorithm should choose the path around the 10s.
        path = astar.find_path((0, 0), (2, 4))
        
        assert path is not None
        total_cost = calculate_path_cost(grid, path)
        
        # The optimal path should avoid the 10s entirely.
        # A path going (0,0)->(0,4)->(2,4) or similar.
        # Minimum steps: 4 right + 2 down = 6 steps. 7 cells total.
        # Cost = 7 * 1 = 7.
        # If it went through 10s, cost would be significantly higher.
        assert total_cost < 20, f"Path cost {total_cost} is too high, likely went through 10s."
        
        # Verify no 10s in path
        for r, c in path:
            assert grid[r][c] != 10

    def test_no_path_exists(self):
        """Test 4: No path exists because the destination is fully blocked."""
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 1]
        ]
        astar = AStarGrid(grid)
        # Start (0,0) to End (2,2). The column 1 is blocked, and row 2 col 1 is blocked.
        # Actually, let's make it fully blocked:
        grid_blocked = [
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1]
        ]
        astar_blocked = AStarGrid(grid_blocked)
        path = astar_blocked.find_path((0, 0), (0, 2))
        
        assert path is None

    def test_start_equals_end(self):
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
        assert len(path) == 1

    def test_invalid_coordinates(self):
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
            
        # Test negative column
        with pytest.raises(ValueError):
            astar.find_path((0, -1), (0, 0))

    def test_start_or_end_is_wall(self):
        """Additional edge case: Start or end is a wall (0)."""
        grid = [
            [1, 0, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Start is a wall
        path_start_wall = astar.find_path((0, 1), (1, 1))
        assert path_start_wall is None
        
        # End is a wall
        path_end_wall = astar.find_path((0, 0), (0, 1))
        assert path_end_wall is None
```

### How to Run

1.  Save the implementation code as `a_star_grid.py`.
2.  Save the test code as `test_a_star_grid.py`.
3.  Ensure `pytest` is installed (`pip install pytest`).
4.  Run the tests from the terminal:
    ```bash
    pytest test_a_star_grid.py -v
    ```

### Key Implementation Details

*   **Cost Logic**: The cost to enter a cell is the value of that cell in the grid. The start cell's cost is included in the total path cost (as per standard grid pathfinding where you "enter" the start cell).
*   **Heuristic**: Manhattan distance is used. Since movement is restricted to 4 directions and costs are positive, this is an admissible heuristic (never overestimates the true cost), ensuring optimality.
*   **Heap Usage**: `heapq` is used with a tuple `(f_score, counter, row, col)`. The `counter` is crucial to prevent Python from trying to compare tuples of `(row, col)` when `f_score` values are equal, which would raise a `TypeError` if the coordinates are not comparable in a specific way or to ensure deterministic tie-breaking.
*   **Edge Cases**:
    *   **Start == End**: Returns immediately with a list containing just the start coordinate.
    *   **Walls**: If start or end is `0`, returns `None`.
    *   **Bounds**: Raises `ValueError` if coordinates are outside the grid dimensions.
*   **Optimality**: The algorithm uses the standard A* logic: `f = g + h`. It only updates a node if a strictly lower `g` (cost from start) is found, guaranteeing the shortest path in terms of total cost.
