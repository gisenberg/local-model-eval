# Qwen3.5-9B (L40S vLLM) — A* Pathfinding — Run 2

## Output

Here is the complete implementation of the `AStarGrid` class followed by the `pytest` test suite.

### 1. Implementation: `astar_grid.py`

```python
import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.
    
    Grid values represent movement cost:
    - 0: Impassable wall
    - Positive int: Cost to enter that cell
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.
        
        Args:
            grid: 2D list where grid[r][c] is the cost to enter cell (r, c).
                  0 represents a wall.
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
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self._is_valid(nr, nc):
                # Check if not a wall (cost > 0)
                if self.grid[nr][nc] != 0:
                    neighbors.append((nr, nc))
        return neighbors

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance heuristic.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(
        self, 
        start: Tuple[int, int], 
        end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A*.
        
        Args:
            start: Tuple (row, col) of the starting cell.
            end: Tuple (row, col) of the target cell.
            
        Returns:
            List of (row, col) tuples representing the path from start to end inclusive.
            Returns None if no path exists.
            Raises ValueError if coordinates are out of bounds.
        """
        # Validate bounds
        if not self._is_valid(start[0], start[1]):
            raise ValueError(f"Start coordinates {start} are out of bounds.")
        if not self._is_valid(end[0], end[1]):
            raise ValueError(f"End coordinates {end} are out of bounds.")

        # Validate walls
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None

        # Edge case: Start is the same as End
        if start == end:
            return [start]

        # Priority Queue: (f_score, counter, current_node)
        # Counter is used to break ties in f_score to ensure FIFO order for same f_score
        counter = 0
        open_set = [(self._heuristic(start, end), counter, start)]
        came_from = {}
        g_score = {start: 0}
        
        # Track visited nodes to avoid re-processing in a way that breaks optimality
        # In standard A*, we check if we found a better path to a node before expanding.
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            # Explore neighbors
            for neighbor in self._get_neighbors(current[0], current[1]):
                neighbor_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + neighbor_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return None
```

### 2. Tests: `test_astar_grid.py`

```python
import pytest
from typing import List, Tuple
from astar_grid import AStarGrid

def get_path_cost(path: List[Tuple[int, int]], grid: List[List[int]]) -> int:
    """Helper to calculate total cost of a path."""
    if not path:
        return 0
    # Cost is sum of entering cells. Start cell cost is usually ignored in movement cost 
    # unless specified, but based on problem description "cost to enter that cell",
    # we sum costs of all cells in the path. 
    # Note: If start cost is meant to be excluded, subtract grid[path[0]].
    # Standard interpretation: Sum of weights of all nodes in path.
    return sum(grid[r][c] for r, c in path)

class TestAStarGrid:
    
    @pytest.fixture
    def uniform_grid(self) -> List[List[int]]:
        # 3x3 grid, all cost 1
        return [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]

    @pytest.fixture
    def obstacle_grid(self) -> List[List[int]]:
        # 3x3 grid with a wall in the middle
        return [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]

    @pytest.fixture
    def weighted_grid(self) -> List[List[int]]:
        # Grid with varying costs
        # Start (0,0) -> End (2,2)
        # Path 1 (Right then Down): (0,1)[10] + (0,2)[10] + (1,2)[10] + (2,2)[10] = 40
        # Path 2 (Down then Right): (1,0)[1] + (2,0)[1] + (2,1)[1] + (2,2)[1] = 4
        return [
            [1, 10, 10],
            [1, 1, 1],
            [1, 1, 1]
        ]

    def test_simple_path_uniform_grid(self, uniform_grid: List[List[int]]):
        """Test simple path on a uniform cost grid."""
        grid = AStarGrid(uniform_grid)
        start = (0, 0)
        end = (2, 2)
        
        path = grid.find_path(start, end)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == end
        assert len(path) == 5  # (0,0)->(0,1)->(0,2)->(1,2)->(2,2) or similar
        
        # Verify optimality (Manhattan distance * cost = 4 * 1 = 4 steps + start? 
        # Actually path length in nodes is 5. Sum of costs: 1+1+1+1+1 = 5)
        # Wait, logic check: 
        # Path: (0,0), (0,1), (0,2), (1,2), (2,2). Costs: 1+1+1+1+1 = 5.
        # Any other path of length 4 moves has 5 nodes.
        total_cost = get_path_cost(path, uniform_grid)
        assert total_cost == 5

    def test_path_around_obstacles(self, obstacle_grid: List[List[int]]):
        """Test finding a path that goes around a wall."""
        grid = AStarGrid(obstacle_grid)
        start = (0, 0)
        end = (2, 2)
        
        path = grid.find_path(start, end)
        
        assert path is not None
        # The direct center path is blocked, must go around.
        # Likely path: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) OR (0,0)->(1,0)->(2,0)->(2,1)->(2,2)
        # Both have cost 5.
        assert path[0] == start
        assert path[-1] == end
        assert get_path_cost(path, obstacle_grid) == 5

    def test_weighted_grid_optimality(self, weighted_grid: List[List[int]]):
        """Test that the algorithm prefers lower-cost cells in a weighted grid."""
        grid = AStarGrid(weighted_grid)
        start = (0, 0)
        end = (2, 2)
        
        path = grid.find_path(start, end)
        
        assert path is not None
        # Expected optimal path: Down, Down, Right, Right
        # Nodes: (0,0), (1,0), (2,0), (2,1), (2,2)
        # Costs: 1, 1, 1, 1, 1 -> Total 5
        # Alternative (Right first): (0,0), (0,1), (0,2), (1,2), (2,2)
        # Costs: 1, 10, 10, 10, 10 -> Total 41
        
        expected_path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
        assert path == expected_path
        
        total_cost = get_path_cost(path, weighted_grid)
        assert total_cost == 5

    def test_no_path_exists(self, obstacle_grid: List[List[int]]):
        """Test case where no path exists (fully blocked or disconnected)."""
        grid = AStarGrid(obstacle_grid)
        # Create a scenario where start and end are separated by walls completely
        # Modify grid temporarily for this specific test logic or use a new grid
        blocked_grid = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ]
        grid_blocked = AStarGrid(blocked_grid)
        
        start = (0, 0)
        end = (2, 2)
        
        path = grid_blocked.find_path(start, end)
        assert path is None

    def test_start_equals_end(self, uniform_grid: List[List[int]]):
        """Test edge case where start and end are the same."""
        grid = AStarGrid(uniform_grid)
        start = (1, 1)
        
        path = grid.find_path(start, start)
        
        assert path is not None
        assert path == [start]
        assert len(path) == 1

    def test_invalid_coordinates(self, uniform_grid: List[List[int]]):
        """Test that invalid coordinates raise ValueError."""
        grid = AStarGrid(uniform_grid)
        
        # Out of bounds start
        with pytest.raises(ValueError):
            grid.find_path((-1, 0), (0, 0))
            
        # Out of bounds end
        with pytest.raises(ValueError):
            grid.find_path((0, 0), (10, 10))
            
        # Start is a wall
        wall_grid = [[0, 1], [1, 1]]
        grid_wall = AStarGrid(wall_grid)
        assert grid_wall.find_path((0, 0), (0, 1)) is None
        
        # End is a wall
        assert grid_wall.find_path((0, 1), (0, 0)) is None
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
*   **Data Structures**:
    *   `heapq` is used for the open set to efficiently retrieve the node with the lowest `f_score`.
    *   `g_score` dictionary tracks the actual cost from start to current node.
    *   `came_from` dictionary allows reconstruction of the path once the goal is reached.
*   **Optimality**: The algorithm guarantees the shortest path (minimum total cost) because A* with an admissible heuristic (Manhattan) on a graph with non-negative edge weights is optimal.
*   **Edge Cases**:
    *   Explicitly checks if start/end are walls or out of bounds before running the search.
    *   Handles `start == end` immediately to return `[start]`.
    *   Returns `None` if the queue empties without reaching the goal.
