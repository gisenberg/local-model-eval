# Qwen 3.5 9B 4bit MLX — astar

## Output

Here is the complete implementation of the A* pathfinding algorithm for a weighted 2D grid, followed by the required pytest test suite.

### 1. Implementation (`astar_grid.py`)

```python
import heapq
from typing import List, Optional, Tuple, Dict

class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.
    
    The grid uses 0 to represent impassable walls and positive integers 
    to represent the cost to enter a specific cell.
    
    Movement is restricted to 4 directions (Up, Down, Left, Right).
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.
        
        Args:
            grid: A 2D list of integers. 0 = wall, >0 = cost.
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
        Get valid 4-directional neighbors.
        Returns a list of (row, col) tuples for valid neighbors.
        """
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self._is_valid(nr, nc):
                neighbors.append((nr, nc))
        return neighbors

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """
        Calculate Manhattan distance heuristic.
        h(n) = |n.row - goal.row| + |n.col - goal.col|
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Tuple (row, col) of the starting cell.
            end: Tuple (row, col) of the destination cell.
            
        Returns:
            A list of tuples representing the path coordinates, or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate start and end positions
        if not self._is_valid(start[0], start[1]):
            raise ValueError(f"Start position {start} is out of bounds.")
        if not self._is_valid(end[0], end[1]):
            raise ValueError(f"End position {end} is out of bounds.")

        # Check if start is a wall
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall (0).")
            
        # Check if end is a wall
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall (0).")

        # Handle start == end case
        if start == end:
            return [start]

        # Priority Queue: (f_score, counter, node)
        # Counter is used to break ties in f_score to ensure deterministic behavior
        # and to handle cases where multiple nodes have the same f_score.
        open_set = []
        counter = 0
        
        # g_score: dictionary to store the cost of reaching a node
        g_score = {start: 0}
        
        # f_score: f = g + h
        f_score = {start: self._heuristic(start, end)}
        
        heapq.heappush(open_set, (f_score[start], counter, start))
        counter += 1
        
        came_from = {}
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            current_node = current
            
            # If we reached the goal
            if current_node == end:
                # Reconstruct path
                total_path = [end]
                while current_node in came_from:
                    current_node = came_from[current_node]
                    total_path.append(current_node)
                total_path.reverse()
                return total_path
            
            # Current g_score might be outdated if this node was processed before
            if current_node in g_score:
                current_g = g_score[current_node]
            else:
                continue # Skip if not in g_score (already processed with better path)
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current_node[0], current_node[1]):
                # Check if neighbor is a wall
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue
                
                # Calculate tentative g_score
                movement_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = current_g + movement_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    
                    if neighbor not in open_set:
                        heapq.heappush(open_set, (f, counter, neighbor))
                        counter += 1
                    else:
                        # Update existing entry in heap (lazy deletion handled by check above)
                        # We update the priority in the heap by pushing a new entry
                        # The old entry will be ignored when popped if its g_score is worse
                        heapq.heappush(open_set, (f, counter, neighbor))
                        counter += 1
                    
                    came_from[neighbor] = current_node

        return None
```

### 2. Pytest Tests (`test_astar_grid.py`)

```python
import pytest
from typing import List, Tuple, Optional
from astar_grid import AStarGrid

class TestAStarGrid:
    """Test suite for AStarGrid class."""

    def test_simple_path(self):
        """Test finding a simple path in an empty grid."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 2)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        assert len(path) == 5  # (0,0)->(0,1)->(0,2)->(1,2)->(2,2) or similar optimal
        assert path[0] == start
        assert path[-1] == end

    def test_start_equals_end(self):
        """Test case where start and end are the same."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (0, 0)
        
        path = astar.find_path(start, end)
        
        assert path == [start]

    def test_no_path_walls(self):
        """Test when no path exists due to walls."""
        grid = [
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 2)
        
        path = astar.find_path(start, end)
        
        assert path is None

    def test_out_of_bounds_start(self):
        """Test ValueError when start is out of bounds."""
        grid = [[1, 1]]
        astar = AStarGrid(grid)
        start = (5, 5)
        end = (0, 0)
        
        with pytest.raises(ValueError, match="Start position.*is out of bounds"):
            astar.find_path(start, end)

    def test_out_of_bounds_end(self):
        """Test ValueError when end is out of bounds."""
        grid = [[1, 1]]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (5, 5)
        
        with pytest.raises(ValueError, match="End position.*is out of bounds"):
            astar.find_path(start, end)

    def test_start_is_wall(self):
        """Test ValueError when start position is a wall."""
        grid = [
            [0, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (1, 1)
        
        with pytest.raises(ValueError, match="Start position is a wall"):
            astar.find_path(start, end)

    def test_optimal_weighted_path(self):
        """Test that the path found is optimal based on weighted costs."""
        # Grid layout:
        # 1 10 1
        # 1  1 1
        # 1  1 1
        # Path A: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) Cost: 1+10+1+1+1 = 14
        # Path B: (0,0)->(1,0)->(2,0)->(2,1)->(2,2) Cost: 1+1+1+1+1 = 5
        # Path C: (0,0)->(1,0)->(1,1)->(1,2)->(2,2) Cost: 1+1+1+1+1 = 5 (Same as B)
        # Note: (1,1) is cost 1, (0,1) is cost 10. A* should avoid (0,1).
        
        grid = [
            [1, 10, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 2)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        
        # Calculate total cost of the returned path
        total_cost = 0
        current = start
        # Skip start cost calculation logic if needed, but usually path includes entry costs.
        # Let's verify the path goes through the cheap route (down then right).
        # Expected path: (0,0), (1,0), (2,0), (2,1), (2,2)
        expected_path = [(0,0), (1,0), (2,0), (2,1), (2,2)]
        
        assert path == expected_path
        
        # Verify cost calculation manually
        costs = [grid[r][c] for r, c in path]
        total_cost = sum(costs)
        assert total_cost == 5  # 1+1+1+1+1

    def test_optimal_weighted_path_complex(self):
        """Test optimal path selection in a more complex weighted grid."""
        # Grid:
        # 1 100 1
        # 1  1 1
        # 1  1 1
        # 1  1 1
        # Start (0,0), End (4,2)
        # Route 1 (Top): 1 + 100 + 1 + 1 + 1 + 1 + 1 = 106
        # Route 2 (Bottom): 1 + 1 + 1 + 1 + 1 + 1 + 1 = 7
        # A* should choose Route 2.
        
        grid = [
            [1, 100, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (3, 2) # Changed end to (3,2) to make it fit 4 rows
        
        # Let's adjust grid to 4x3
        grid = [
            [1, 100, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        start = (0, 0)
        end = (3, 2)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        
        # Check if it went down the left side (avoiding 100)
        # Path should be: (0,0)->(1,0)->(2,0)->(3,0)->(3,1)->(3,2)
        expected_path = [(0,0), (1,0), (2,0), (3,0), (3,1), (3,2)]
        assert path == expected_path
```

### How to Run

1.  Save the first code block as `astar_grid.py`.
2.  Save the second code block as `test_astar_grid.py`.
3.  Run the tests using pytest:
    ```bash
    pytest test_astar_grid.py -v
    ```

### Key Features Explained

1.  **Weighted Grid**: The `grid` values are treated as entry costs. If a cell is `0`, it is treated as a wall.
2.  **4-Directional Movement**: The `_get_neighbors` method checks Up, Down, Left, and Right.
3.  **Manhattan Heuristic**: Used for the `h(n)` component. It is admissible for 4-directional movement on a grid (never overestimates the true cost).
4.  **Tie-Breaking**: The priority queue stores tuples `(f_score, counter, node)`. The `counter` increments every time a node is added to the heap. This ensures that if two nodes have the same `f_score`, the one added earlier (or processed in a specific order) is popped first, preventing infinite loops in edge cases and ensuring deterministic behavior.
5.  **Optimality**: The algorithm maintains a `g_score` dictionary. If a node is reached via a path with a lower `g_score` than previously recorded, the `f_score` is updated, and the node is re-added to the heap. This guarantees the first time the goal is popped from the heap, it is via the optimal path.
6.  **Error Handling**:
    *   `ValueError` is raised if the grid is empty.
    *   `ValueError` is raised if `start` or `end` coordinates are outside the grid dimensions.
    *   `ValueError` is raised if `start` or `end` positions contain a `0` (wall).
7.  **Edge Case `start == end`**: Explicitly handled to return `[start]` immediately without running the search loop.
