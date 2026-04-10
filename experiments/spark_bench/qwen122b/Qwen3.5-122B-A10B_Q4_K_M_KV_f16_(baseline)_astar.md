# Qwen3.5-122B-A10B Q4_K_M — KV f16 (baseline) — A* Pathfinding

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
        Initializes the AStarGrid with the provided grid.
        
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
            List of valid (row, col) tuples that are within bounds and not walls.
        """
        r, c = pos
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] != 0: # 0 is a wall
                    neighbors.append((nr, nc))
        
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from start to end using the A* algorithm.
        
        Args:
            start: Tuple (row, col) representing the starting position.
            end: Tuple (row, col) representing the target position.
            
        Returns:
            A list of (row, col) tuples representing the path from start to end (inclusive),
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start coordinates {start} are out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End coordinates {end} are out of bounds.")

        # Edge case: Start or End is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
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
            
            # If we reached the goal, reconstruct path
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

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

        return None
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
        # Cost = 1 (start) + 1 + 1 + 1 + 1 + 1 (end) = 6? 
        # Wait, path includes start and end. 
        # Path length (nodes) = 5 (e.g., (0,0)->(0,1)->(0,2)->(1,2)->(2,2))
        # Cost = 1+1+1+1+1 = 5.
        # Let's verify length.
        assert len(path) == 5
        assert calculate_path_cost(grid, path) == 5

    def test_path_around_obstacles(self):
        """Test 2: Path finding around obstacles (0s)."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 3))
        
        assert path is not None
        # The path must go around the middle block
        # One valid path: (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3)
        # Another: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(2,3)
        # Both have length 6 nodes, cost 6.
        assert len(path) == 6
        assert calculate_path_cost(grid, path) == 6
        # Verify no walls are in path
        for r, c in path:
            assert grid[r][c] != 0

    def test_weighted_grid_optimality(self):
        """Test 3: Weighted grid where path prefers lower-cost cells."""
        # High cost in the middle row, low cost on top/bottom
        grid = [
            [1, 1, 1, 1],
            [1, 10, 10, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 3))
        
        assert path is not None
        
        # Direct path through middle would be: (0,0)->(1,0)->(1,1)->(1,2)->(1,3)->(2,3)
        # Cost: 1 + 1 + 10 + 10 + 1 + 1 = 24
        # Path around top/bottom: (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3)
        # Cost: 1 + 1 + 1 + 1 + 1 + 1 = 6
        
        assert calculate_path_cost(grid, path) == 6
        
        # Verify the path does not go through the high cost cells (10)
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
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Start is wall
        assert astar.find_path((0, 0), (2, 2)) is None
        
        # End is wall
        assert astar.find_path((0, 1), (0, 0)) is None
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

### Key Implementation Details

*   **Heuristic**: Uses Manhattan distance (`|x1-x2| + |y1-y2|`), which is admissible for 4-directional movement on a grid.
*   **Optimality**: The algorithm uses a priority queue (min-heap) ordered by $f(n) = g(n) + h(n)$. Since the heuristic is admissible and the edge weights are non-negative, A* guarantees the optimal path.
*   **Cost Calculation**: The cost of a path is the sum of the values of the cells entered, including the start and end cells.
*   **Data Structures**:
    *   `open_set`: A min-heap storing `(f_score, counter, row, col)`. The `counter` is used to break ties in the heap to ensure deterministic behavior when `f_score` is equal.
    *   `g_score`: A dictionary mapping coordinates to the lowest known cost from the start.
    *   `came_from`: A dictionary to reconstruct the path once the goal is reached.
*   **Edge Cases**:
    *   **Start == End**: Returns immediately with a list containing just the start coordinate.
    *   **Walls**: If start or end is `0`, returns `None`.
    *   **Bounds**: Raises `ValueError` if coordinates are outside the grid dimensions.
