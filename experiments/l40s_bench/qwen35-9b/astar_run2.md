# Qwen3.5-9B (L40S vLLM) — A* Pathfinding — Run 2

## Output

Here is the complete implementation of the `AStarGrid` class and the corresponding `pytest` test suite.

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
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if self._is_valid(nr, nc) and self.grid[nr][nc] != 0:
                neighbors.append((nr, nc))
        return neighbors

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance heuristic.
        
        Args:
            a: Current cell (row, col).
            b: Target cell (row, col).
            
        Returns:
            Manhattan distance: |r1-r2| + |c1-c2|.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting coordinates (row, col).
            end: Target coordinates (row, col).
            
        Returns:
            List of coordinates representing the path from start to end inclusive.
            Returns None if no path exists.
            Raises ValueError if start or end are out of bounds.
        """
        # Validate bounds
        if not self._is_valid(start[0], start[1]):
            raise ValueError(f"Start position {start} is out of bounds.")
        if not self._is_valid(end[0], end[1]):
            raise ValueError(f"End position {end} is out of bounds.")

        # Validate walls
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None

        # Edge case: Start equals End
        if start == end:
            return [start]

        # Priority Queue: (f_score, counter, g_score, position)
        # Counter is used to break ties in f_score to ensure deterministic behavior
        # and to handle the case where multiple nodes have same f_score.
        counter = 0
        open_set = []
        heapq.heappush(open_set, (0, counter, 0, start))
        
        # Track visited nodes and their best known g_score
        # Using a dictionary for sparse storage
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, end)}

        while open_set:
            _, _, current_g, current = heapq.heappop(open_set)

            # If we reached the target
            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            # If we found a shorter path to this node already, skip
            if current_g > g_score.get(current, float('inf')):
                continue

            # Explore neighbors
            for neighbor in self._get_neighbors(current[0], current[1]):
                neighbor_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + neighbor_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    f_score[neighbor] = f
                    counter += 1
                    heapq.heappush(open_set, (f, counter, tentative_g, neighbor))

        return None  # No path found
```

### 2. Tests: `test_astar_grid.py`

```python
import pytest
from typing import List, Tuple
from astar_grid import AStarGrid

def calculate_total_cost(path: List[Tuple[int, int]], grid: List[List[int]]) -> int:
    """Helper to calculate total cost of a path."""
    if not path:
        return 0
    # Cost to enter the first cell
    cost = grid[path[0][0]][path[0][1]]
    # Cost to enter subsequent cells
    for r, c in path[1:]:
        cost += grid[r][c]
    return cost

class TestAStarGrid:
    
    def test_simple_path_uniform_grid(self):
        """Test 1: Simple path on a uniform grid (no obstacles)."""
        grid = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 3)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == end
        assert len(path) == 5  # (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3) is one option, length 5
        
        # Verify optimality: Manhattan distance is 5. Since all costs are 1, min cost is 5.
        total_cost = calculate_total_cost(path, grid)
        assert total_cost == 5

    def test_path_around_obstacles(self):
        """Test 2: Path around obstacles."""
        grid = [
            [1, 0, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (0, 3)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        # The direct path is blocked at (0,1). Must go down to row 1.
        # Path: (0,0)->(1,0)->(1,1)->(1,2)->(1,3)->(0,3)
        assert len(path) == 6
        assert path[0] == start
        assert path[-1] == end
        
        # Verify optimality: 6 cells * cost 1 = 6
        total_cost = calculate_total_cost(path, grid)
        assert total_cost == 6

    def test_weighted_grid_prefers_lower_cost(self):
        """Test 3: Weighted grid where path prefers lower-cost cells."""
        # Grid layout:
        # 1  10  1
        # 1  1   1
        # 1  1   1
        # Direct top row is expensive (10). Bottom row is cheap.
        grid = [
            [1, 10, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (0, 2)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        # Expected path goes down, across bottom, up: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2)
        # Length: 7 nodes
        assert len(path) == 7
        
        # Verify optimality: 7 cells * cost 1 = 7
        # If it took the top path: 1 + 10 + 1 = 12
        total_cost = calculate_total_cost(path, grid)
        assert total_cost == 7

    def test_no_path_exists_fully_blocked(self):
        """Test 4: No path exists (fully blocked)."""
        grid = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (0, 2)
        
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
        # Cost is just the cost of entering the start cell
        assert calculate_total_cost(path, grid) == 1

    def test_invalid_coordinates(self):
        """Test 6: Invalid coordinates (out of bounds)."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Test out of bounds start
        with pytest.raises(ValueError):
            astar.find_path((5, 5), (0, 0))
            
        # Test out of bounds end
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (5, 5))
            
        # Test start is a wall
        grid_wall = [
            [0, 1],
            [1, 1]
        ]
        astar_wall = AStarGrid(grid_wall)
        result = astar_wall.find_path((0, 0), (0, 1))
        assert result is None
        
        # Test end is a wall
        result = astar_wall.find_path((0, 1), (0, 0))
        assert result is None
```

### How to Run

1.  Save the first code block as `astar_grid.py`.
2.  Save the second code block as `test_astar_grid.py`.
3.  Run the tests using pytest:
    ```bash
    pytest test_astar_grid.py -v
    ```

### Key Features Explained

*   **Optimality**: The algorithm uses `g_score` (actual cost from start) and `f_score` (g_score + heuristic). By always expanding the node with the lowest `f_score` via a min-heap, it guarantees the path with the minimum total cost is found first.
*   **Heuristic**: Manhattan distance (`|x1-x2| + |y1-y2|`) is admissible for 4-directional movement on a grid with non-negative weights, ensuring A* finds the optimal path.
*   **Tie-Breaking**: A `counter` is added to the heap tuple `(f_score, counter, g_score, pos)`. This ensures that if two nodes have the same `f_score`, the one discovered earlier (or processed in a consistent order) is popped first, preventing infinite loops in edge cases and ensuring deterministic behavior.
*   **Edge Cases**:
    *   **Walls**: Checked immediately upon initialization of start/end and during neighbor expansion.
    *   **Bounds**: Checked immediately at the start of `find_path`.
    *   **Start == End**: Handled explicitly to return `[start]` with O(1) complexity.
    *   **No Path**: The loop terminates when the open set is empty, returning `None`.
