# Qwen3.5-9B (L40S vLLM) — A* Pathfinding — Run 3

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
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.
        
        Args:
            grid: A 2D list where grid[r][c] is the cost to enter cell (r, c).
                  0 represents a wall (impassable).
        """
        self.grid = grid
        self.rows = len(grid)
        if self.rows == 0:
            self.cols = 0
        else:
            self.cols = len(grid[0])

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """
        Get valid 4-directional neighbors (up, down, left, right).
        Excludes walls (cost 0).
        """
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self._is_valid(nr, nc) and self.grid[nr][nc] != 0:
                neighbors.append((nr, nc))
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Tuple (row, col) of the starting cell.
            end: Tuple (row, col) of the target cell.
            
        Returns:
            A list of (row, col) tuples representing the path from start to end inclusive.
            Returns None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate bounds
        if not self._is_valid(start[0], start[1]):
            raise ValueError(f"Start coordinates {start} are out of bounds.")
        if not self._is_valid(end[0], end[1]):
            raise ValueError(f"End coordinates {end} are out of bounds.")
        
        sr, sc = start
        er, ec = end

        # Edge case: Start is a wall
        if self.grid[sr][sc] == 0:
            return None
        # Edge case: End is a wall
        if self.grid[er][ec] == 0:
            return None

        # Edge case: Start equals End
        if start == end:
            return [start]

        # Heuristic: Manhattan distance
        def heuristic(r: int, c: int) -> int:
            return abs(r - er) + abs(c - ec)

        # Priority Queue: (f_score, g_score, r, c)
        # f_score = g_score + heuristic
        open_set = []
        heapq.heappush(open_set, (heuristic(sr, sc), 0, sr, sc))
        
        # Track best known g_score (cost to reach cell)
        # Initialize with infinity
        g_score = {(sr, sc): 0}
        
        # To reconstruct path
        came_from = {}

        while open_set:
            _, current_g, curr_r, curr_c = heapq.heappop(open_set)

            # If we reached the goal
            if (curr_r, curr_c) == (er, ec):
                # Reconstruct path
                path = []
                current = (curr_r, curr_c)
                while current in came_from or current == start:
                    path.append(current)
                    if current == start:
                        break
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            # If we found a better way to this node already, skip
            if current_g > g_score.get((curr_r, curr_c), float('inf')):
                continue

            # Explore neighbors
            for nr, nc in self._get_neighbors(curr_r, curr_c):
                neighbor_cost = self.grid[nr][nc]
                tentative_g = current_g + neighbor_cost

                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = (curr_r, curr_c)
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + heuristic(nr, nc)
                    heapq.heappush(open_set, (f_score, tentative_g, nr, nc))

        return None
```

### 2. Tests: `test_astar_grid.py`

```python
import pytest
from typing import List, Tuple, Optional
from astar_grid import AStarGrid

def calculate_path_cost(path: List[Tuple[int, int]], grid: List[List[int]]) -> int:
    """Helper to calculate total cost of a path for assertion."""
    if not path:
        return 0
    total_cost = grid[path[0][0]][path[0][1]] # Cost to enter start
    for r, c in path[1:]:
        total_cost += grid[r][c]
    return total_cost

class TestAStarGrid:
    
    def test_simple_path_uniform_grid(self):
        """Test 1: Simple path on a uniform grid."""
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
        assert len(path) == 4  # Direct diagonal-like shortest path in grid steps
        
        # Optimal cost: 4 steps * cost 1 = 4
        assert calculate_path_cost(path, grid) == 4

    def test_path_around_obstacles(self):
        """Test 2: Path finding around obstacles."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (3, 3)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        # Path should go down, right, right, down, down, right, right (or similar valid detour)
        # Expected cost: 1 (start) + 1 + 1 + 1 + 1 + 1 + 1 = 7 (going around the 2x2 hole)
        # Actually: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(2,3)->(3,3) is 7 steps.
        # Or (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3)->(3,3) is 7 steps.
        assert calculate_path_cost(path, grid) == 7

    def test_weighted_grid_optimality(self):
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
        # Optimal path: (0,0) -> (1,0) -> (1,1) -> (1,2) -> (2,2)
        # Cost: 1 + 1 + 1 + 1 + 1 = 5
        # Alternative (via top right): 1 + 10 + 1 + 10 + 1 = 23
        assert calculate_path_cost(path, grid) == 5
        assert path == [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)]

    def test_no_path_exists(self):
        """Test 4: No path exists (fully blocked)."""
        grid = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 2)
        
        path = astar.find_path(start, end)
        
        assert path is None

    def test_start_equals_end(self):
        """Test 5: Start equals end."""
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
            astar.find_path((-1, 0), (0, 0))
            
        # Out of bounds end
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (5, 5))
            
        # Start is a wall
        grid_with_wall = [[0, 1], [1, 1]]
        astar_wall = AStarGrid(grid_with_wall)
        path = astar_wall.find_path((0, 0), (1, 1))
        assert path is None
        
        # End is a wall
        grid_wall_end = [[1, 1], [1, 0]]
        astar_wall_end = AStarGrid(grid_wall_end)
        path = astar_wall_end.find_path((0, 0), (1, 1))
        assert path is None
```

### How to Run

1.  Save the first code block as `astar_grid.py`.
2.  Save the second code block as `test_astar_grid.py`.
3.  Run the tests using pytest:
    ```bash
    pytest test_astar_grid.py -v
    ```

### Key Features Explained

*   **Optimality**: The algorithm maintains a `g_score` dictionary to track the lowest cost found so far to reach any specific cell. If a cheaper path to a cell is found later, the priority queue is updated, ensuring the first time we pop the goal node, it is via the optimal path.
*   **Heuristic**: Uses Manhattan distance (`|r1-r2| + |c1-c2|`), which is admissible for 4-directional movement on a grid, guaranteeing optimality.
*   **Edge Cases**:
    *   **Walls**: Cells with cost `0` are treated as walls. The `_get_neighbors` method explicitly filters them out.
    *   **Start/End Walls**: Checked immediately after bounds validation.
    *   **Bounds**: Explicitly checked and raise `ValueError` as requested.
    *   **Start == End**: Handled as a special case to return `[start]` immediately with cost equal to the cell's entry cost.
*   **Data Structures**: Uses `heapq` for the open set (min-heap) and dictionaries for `g_score` and `came_from` to ensure O(1) lookups and path reconstruction.
