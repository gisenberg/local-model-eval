# Qwen3.5-122B-A10B Q4_K_M (bartowski) — KV f16 (baseline) — A* Pathfinding

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
        Initializes the AStarGrid with the provided 2D grid.
        
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
                raise ValueError("All rows in the grid must have the same length.")

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculates the Manhattan distance between two points.
        
        Args:
            a: Tuple (row, col) for the first point.
            b: Tuple (row, col) for the second point.
            
        Returns:
            The Manhattan distance (|x1-x2| + |y1-y2|).
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Returns a list of valid 4-directional neighbors for a given position.
        
        Args:
            pos: Tuple (row, col) representing the current position.
            
        Returns:
            List of valid (row, col) tuples for neighbors.
        """
        r, c = pos
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                # Check if the cell is not a wall (cost > 0)
                if self.grid[nr][nc] > 0:
                    neighbors.append((nr, nc))
        
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from start to end using the A* algorithm.
        
        Args:
            start: Tuple (row, col) representing the starting position.
            end: Tuple (row, col) representing the target position.
            
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

        # Check if start or end are walls
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None

        # Edge case: Start equals End
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
                # Cost to enter the neighbor cell
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    f_score[neighbor] = f
                    
                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f, counter, neighbor[0], neighbor[1]))
                        open_set_hash.add(neighbor)

        # No path found
        return None

    def _reconstruct_path(self, came_from: dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstructs the path from start to end by backtracking through came_from.
        
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
        
        # Optimal cost for 3x3 grid from corner to corner is 5 steps (Manhattan distance)
        # Cost = 1 * 5 = 5 (since we count the cost of the start cell too? 
        # Note: Usually path cost includes start and end. 
        # Path length is 5 cells: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) is 5 cells? 
        # Actually Manhattan distance is 4 steps, so 5 cells.
        # Let's verify: (0,0) to (2,2) requires 2 down, 2 right. Total 4 moves. 5 cells visited.
        # Cost = 5 * 1 = 5.
        assert calculate_path_cost(grid, path) == 5

    def test_path_around_obstacles(self):
        """Test 2: Path finding around obstacles (walls)."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        # Start (0,0) to End (2,3). Must go around the wall in row 1.
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
        # Direct path through middle: 1 + 10 + 10 + 10 + 1 = 32 (approx, depends on exact cells)
        # Path going around top/bottom: 
        # (0,0)->(0,1)->(0,2)->(0,3)->(0,4)->(1,4)->(2,4) 
        # Cost: 1+1+1+1+1+1+1 = 7
        path = astar.find_path((0, 0), (2, 4))
        
        assert path is not None
        total_cost = calculate_path_cost(grid, path)
        
        # The optimal path should avoid the 10s.
        # Minimum possible cost is 7 (7 cells of cost 1).
        assert total_cost == 7
        
        # Verify the path does not touch the 10s
        for r, c in path:
            assert grid[r][c] != 10

    def test_no_path_exists(self):
        """Test 4: No path exists because the grid is fully blocked."""
        grid = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
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
        assert len(path) == 1
        assert path[0] == (1, 1)
        assert calculate_path_cost(grid, path) == 1

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
            [0, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Start is wall
        assert astar.find_path((0, 0), (1, 1)) is None
        
        # End is wall
        assert astar.find_path((1, 0), (0, 0)) is None
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

*   **Optimality**: The algorithm uses `g_score` (actual cost from start) + `h_score` (Manhattan heuristic) to prioritize nodes. Since the heuristic is admissible (never overestimates the true cost in a grid with non-negative weights), A* guarantees the optimal path.
*   **Weighted Handling**: The cost to move to a neighbor is the value of the neighbor cell (`grid[nr][nc]`). This correctly models the requirement where entering a cell incurs its specific cost.
*   **Heap Usage**: `heapq` is used with a tuple `(f_score, counter, row, col)`. The `counter` is a tie-breaker to ensure stable sorting when `f_score` values are identical, preventing arbitrary reordering that might affect performance or determinism.
*   **Edge Cases**:
    *   **Start == End**: Returns immediately with `[start]`.
    *   **Walls**: Checks `grid[r][c] == 0` before adding to the open set or returning a path.
    *   **Bounds**: Raises `ValueError` immediately if coordinates are invalid.
*   **Type Hints**: Full type hinting is applied to arguments, return types, and internal variables for clarity and IDE support.
