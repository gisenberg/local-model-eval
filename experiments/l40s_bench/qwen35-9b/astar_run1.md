# Qwen3.5-9B (L40S vLLM) — A* Pathfinding — Run 1

## Output

Here is the complete implementation of the `AStarGrid` class followed by the `pytest` test suite.

### 1. Implementation: `astar_grid.py`

```python
import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.
    
    The grid uses 0 for impassable walls and positive integers for movement costs.
    Movement is restricted to 4 directions (up, down, left, right).
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.
        
        Args:
            grid: A 2D list where grid[r][c] is the cost to enter cell (r, c).
                  0 represents a wall (impassable). Positive integers represent costs.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """
        Get valid neighboring cells (up, down, left, right).
        
        Returns:
            List of (row, col) tuples for valid neighbors.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = []
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self._is_valid(nr, nc):
                # 0 is a wall, so we cannot move into it
                if self.grid[nr][nc] != 0:
                    neighbors.append((nr, nc))
        return neighbors

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance heuristic.
        
        Args:
            a: Current cell (row, col).
            b: Target cell (row, col).
            
        Returns:
            Manhattan distance: |a[0] - b[0]| + |a[1] - b[1]|
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting coordinates (row, col).
            end: Target coordinates (row, col).
            
        Returns:
            List of (row, col) coordinates representing the path from start to end inclusive.
            Returns None if no path exists.
            Raises ValueError if start or end are out of bounds.
        """
        # Validate bounds
        if not self._is_valid(start[0], start[1]):
            raise ValueError(f"Start position {start} is out of bounds.")
        if not self._is_valid(end[0], end[1]):
            raise ValueError(f"End position {end} is out of bounds.")

        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None

        # Edge case: Start equals End
        if start == end:
            return [start]

        # Priority Queue: (f_score, counter, g_score, current_node)
        # Counter is used to break ties in f_score to ensure deterministic behavior
        # and to handle the case where f_scores are identical.
        open_set = []
        counter = 0
        
        start_cost = self.grid[start[0]][start[1]]
        g_score = {start: start_cost}
        f_score = {start: start_cost + self._heuristic(start, end)}
        
        heapq.heappush(open_set, (f_score[start], counter, start[0], start[1]))
        counter += 1

        came_from = {}

        while open_set:
            _, _, current_r, current_c = heapq.heappop(open_set)
            current = (current_r, current_c)

            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            # If we found a better path to this node already, skip
            if current in g_score and g_score[current] < g_score.get(current, float('inf')):
                continue

            for neighbor in self._get_neighbors(current_r, current_c):
                neighbor_r, neighbor_c = neighbor
                move_cost = self.grid[neighbor_r][neighbor_c]
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, counter, neighbor_r, neighbor_c))
                    counter += 1

        return None
```

### 2. Tests: `test_astar_grid.py`

```python
import pytest
from typing import List, Tuple, Optional
from astar_grid import AStarGrid

def calculate_path_cost(path: List[Tuple[int, int]], grid: List[List[int]]) -> int:
    """Helper to calculate total cost of a path."""
    if not path:
        return 0
    total_cost = grid[path[0][0]][path[0][1]]
    for r, c in path[1:]:
        total_cost += grid[r][c]
    return total_cost

class TestAStarGrid:
    
    def test_simple_path_uniform_grid(self):
        """Test 1: Simple path on a uniform grid (no obstacles)."""
        grid = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (3, 3)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == end
        assert len(path) == 7  # 4 rows + 3 cols steps
        
        # Verify optimality: Manhattan distance * cost (1) = 6 steps + start cost = 7
        expected_cost = 7
        assert calculate_path_cost(path, grid) == expected_cost

    def test_path_around_obstacles(self):
        """Test 2: Path finding around obstacles (walls)."""
        grid = [
            [1, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (3, 3)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == end
        
        # The path must go around the walls at (0,1) and (2,2)
        # One valid optimal path: (0,0)->(1,0)->(1,1)->(1,2)->(1,3)->(2,3)->(3,3)
        # Cost: 1+1+1+1+1+1+1 = 7
        assert calculate_path_cost(path, grid) == 7

    def test_weighted_grid_prefers_lower_cost(self):
        """Test 3: Weighted grid where path prefers lower-cost cells."""
        # Grid layout:
        # 1 10 1
        # 1  1 1
        # 1 10 1
        # Direct right then down is expensive (10). Going down first is cheap.
        grid = [
            [1, 10, 1],
            [1,  1, 1],
            [1, 10, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 2)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        # Expected path: (0,0) -> (1,0) -> (1,1) -> (1,2) -> (2,2)
        # Avoids the 10s.
        expected_path = [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)]
        assert path == expected_path
        
        # Cost: 1 + 1 + 1 + 1 + 1 = 5
        assert calculate_path_cost(path, grid) == 5

    def test_no_path_exists_fully_blocked(self):
        """Test 4: No path exists (start/end separated by walls)."""
        grid = [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 2)
        
        path = astar.find_path(start, end)
        
        assert path is None

    def test_start_equals_end(self):
        """Test 5: Start position is the same as end position."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        start = (1, 1)
        end = (1, 1)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        assert path == [start]
        assert calculate_path_cost(path, grid) == 1

    def test_invalid_coordinates(self):
        """Test 6: Invalid coordinates raise ValueError."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Out of bounds start
        with pytest.raises(ValueError):
            astar.find_path((5, 5), (0, 0))
            
        # Out of bounds end
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (5, 5))
            
        # Start is a wall
        grid_with_wall = [[0, 1], [1, 1]]
        astar_wall = AStarGrid(grid_with_wall)
        with pytest.raises(ValueError):
            astar_wall.find_path((0, 0), (1, 1))
            
        # End is a wall
        with pytest.raises(ValueError):
            astar_wall.find_path((1, 1), (0, 0))
```

### How to Run

1.  Save the first code block as `astar_grid.py`.
2.  Save the second code block as `test_astar_grid.py`.
3.  Run the tests using pytest:
    ```bash
    pytest test_astar_grid.py -v
    ```

### Key Features Explained

*   **Optimality**: The algorithm uses `g_score` (actual cost from start) and `f_score` (estimated total cost). By always expanding the node with the lowest `f_score` via a min-heap, it guarantees the shortest path in a weighted graph with non-negative edge weights.
*   **Heuristic**: The Manhattan distance `|x1-x2| + |y1-y2|` is admissible for 4-directional movement on a grid, ensuring A* finds the optimal path.
*   **Tie-Breaking**: A `counter` is added to the heap tuple `(f_score, counter, ...)` to ensure deterministic ordering when two nodes have the exact same `f_score`.
*   **Edge Cases**:
    *   **Walls**: Cells with value `0` are treated as impassable.
    *   **Bounds**: Explicit checks raise `ValueError` immediately if coordinates are invalid.
    *   **Start==End**: Returns the single coordinate list immediately.
*   **Type Hints**: Full type hinting is used for arguments, return types, and internal helper functions to ensure code clarity and IDE support.
