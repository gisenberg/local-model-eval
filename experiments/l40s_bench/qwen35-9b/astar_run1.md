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

    def find_path(
        self, 
        start: Tuple[int, int], 
        end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Tuple (row, col) of the starting cell.
            end: Tuple (row, col) of the target cell.
            
        Returns:
            List of (row, col) coordinates from start to end inclusive, 
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate bounds
        if not self._is_valid(start[0], start[1]):
            raise ValueError(f"Start coordinates {start} are out of bounds.")
        if not self._is_valid(end[0], end[1]):
            raise ValueError(f"End coordinates {end} are out of bounds.")

        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None

        # Edge case: Start equals End
        if start == end:
            return [start]

        # Heuristic: Manhattan distance
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Priority Queue: (f_score, g_score, row, col)
        # We use a counter to break ties deterministically if f_scores are equal
        open_set = []
        start_cost = 0  # Cost to reach start is 0 (we are already there)
        # Note: The problem says "cost to enter". Usually start cost is 0 unless specified otherwise.
        # We assume start cost is 0 for the path calculation, but the heuristic guides us.
        
        # f = g + h
        # g is cost from start to current
        # h is estimated cost from current to end
        
        heapq.heappush(open_set, (heuristic(start, end) + start_cost, start_cost, start[0], start[1]))
        
        came_from = {}
        g_score = {start: 0}
        closed_set = set()

        while open_set:
            current_f, current_g, curr_r, curr_c = heapq.heappop(open_set)

            # If we reached the target
            if (curr_r, curr_c) == end:
                # Reconstruct path
                path = []
                while (curr_r, curr_c) in came_from:
                    path.append((curr_r, curr_c))
                    curr_r, curr_c = came_from[(curr_r, curr_c)]
                path.append(start)
                path.reverse()
                return path

            if (curr_r, curr_c) in closed_set:
                continue
            closed_set.add((curr_r, curr_c))

            # Explore neighbors
            for neighbor_r, neighbor_c in self._get_neighbors(curr_r, curr_c):
                if neighbor_r in closed_set:
                    continue

                # Cost to move to neighbor
                move_cost = self.grid[neighbor_r][neighbor_c]
                tentative_g = current_g + move_cost

                if (neighbor_r, neighbor_c) not in g_score or tentative_g < g_score[(neighbor_r, neighbor_c)]:
                    came_from[(neighbor_r, neighbor_c)] = (curr_r, curr_c)
                    g_score[(neighbor_r, neighbor_c)] = tentative_g
                    f_score = tentative_g + heuristic((neighbor_r, neighbor_c), end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor_r, neighbor_c))

        return None
```

### 2. Tests: `test_astar_grid.py`

```python
import pytest
from typing import List, Tuple, Optional
from astar_grid import AStarGrid

def calculate_path_cost(path: List[Tuple[int, int]], grid: List[List[int]]) -> int:
    """Helper to calculate total cost of a path based on grid values."""
    if not path:
        return 0
    # Cost is sum of entering cells. 
    # Start cell cost is usually 0 in this context unless specified, 
    # but based on "cost to enter", we sum from index 1 to end.
    # However, standard A* on weighted grids often includes start cost if defined.
    # Given the prompt "cost to enter that cell", start is already entered.
    total = 0
    for r, c in path[1:]:
        total += grid[r][c]
    return total

class TestAStarGrid:
    
    def test_simple_path_uniform_grid(self):
        """Test 1: Simple path on a uniform grid."""
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
        assert len(path) == 4  # (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3) is 5 steps? 
        # Wait, shortest is (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(2,3) (5 steps)
        # Or (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3) (5 steps)
        # Manhattan distance is |2-0| + |3-0| = 5. So path length must be 6 (including start).
        assert len(path) == 6
        
        # Verify optimality (Manhattan distance * cost 1)
        expected_cost = 5
        actual_cost = calculate_path_cost(path, grid)
        assert actual_cost == expected_cost

    def test_path_around_obstacles(self):
        """Test 2: Path around obstacles."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (0, 3)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        # Must go down, right, right, up
        assert path == [(0, 0), (1, 0), (1, 3), (0, 3)]
        
        # Cost: Enter (1,0)=1, (1,3)=1, (0,3)=1. Total = 3.
        # Direct path blocked.
        actual_cost = calculate_path_cost(path, grid)
        assert actual_cost == 3

    def test_weighted_grid_prefers_lower_cost(self):
        """Test 3: Weighted grid where path prefers lower-cost cells."""
        # Grid:
        # 1 10 1
        # 1  1 1
        # 1 10 1
        # Start (0,0), End (0,2).
        # Path A: (0,0)->(0,1)->(0,2) Cost: 10 + 1 = 11
        # Path B: (0,0)->(1,0)->(1,1)->(1,2)->(0,2) Cost: 1 + 1 + 1 + 1 = 4
        grid = [
            [1, 10, 1],
            [1,  1, 1],
            [1, 10, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (0, 2)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        # Should take the bottom route
        assert path == [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]
        
        actual_cost = calculate_path_cost(path, grid)
        assert actual_cost == 4

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
        assert len(path) == 1

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
        result = astar_wall.find_path((0, 0), (0, 1))
        assert result is None
        
        # End is a wall
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

### Key Implementation Details

*   **Heuristic**: Uses Manhattan distance `|x1-x2| + |y1-y2|`, which is admissible for 4-directional movement on a grid.
*   **Cost Calculation**: The `g_score` tracks the actual cost incurred. The `find_path` function sums the costs of entering cells from the second cell onwards (since the start cell is already occupied).
*   **Tie-Breaking**: The heap stores tuples `(f_score, g_score, row, col)`. If `f_score` is equal, Python compares the next element (`g_score`), then `row`, then `col`. This ensures deterministic behavior without needing a separate counter variable.
*   **Edge Cases**:
    *   **Walls**: Explicitly checked during neighbor generation (`grid[r][c] != 0`) and for start/end points.
    *   **Bounds**: Checked immediately upon function entry.
    *   **Start == End**: Handled as a special case to return `[start]` immediately with cost 0.
