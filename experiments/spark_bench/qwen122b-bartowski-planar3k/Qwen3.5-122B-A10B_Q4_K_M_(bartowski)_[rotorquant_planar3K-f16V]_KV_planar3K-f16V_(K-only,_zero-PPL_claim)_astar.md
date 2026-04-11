# Qwen3.5-122B-A10B Q4_K_M (bartowski) [rotorquant planar3K/f16V] — KV planar3K/f16V (K-only, zero-PPL claim) — A* Pathfinding

## Output

Here is the complete implementation of the A* pathfinding algorithm on a weighted 2D grid, followed by the required pytest test suite.

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
        Initialize the AStarGrid with the provided 2D grid.
        
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
        Calculate the Manhattan distance heuristic between two points.
        
        Args:
            a: Tuple (row, col) for the current node.
            b: Tuple (row, col) for the goal node.
            
        Returns:
            The Manhattan distance between a and b.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid 4-directional neighbors for a given position.
        
        Args:
            pos: Tuple (row, col) representing the current position.
            
        Returns:
            A list of valid (row, col) tuples for neighbors.
        """
        r, c = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
        neighbors = []
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] != 0: # Check if not a wall
                    neighbors.append((nr, nc))
                    
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using the A* algorithm.
        
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
        
        return None

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from start to current node using the came_from map.
        
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
        
        # In a 3x3 uniform grid, shortest path length (nodes) is 5 (e.g., R,R,D,D or D,D,R,R)
        # Cost = 5 * 1 = 5
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
        # Start (0,0) to End (2,3). Must go around the wall in row 1.
        path = astar.find_path((0, 0), (2, 3))
        
        assert path is not None
        # Verify no wall cells are in the path
        for r, c in path:
            assert grid[r][c] != 0
        
        # Verify connectivity
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            assert abs(r1 - r2) + abs(c1 - c2) == 1

    def test_weighted_grid_optimality(self):
        """Test 3: Weighted grid where path prefers lower-cost cells over shorter distance."""
        # Grid:
        # S 1 1
        # 1 10 1
        # 1 1 1 E
        # Direct path via middle (cost 10) is shorter in steps but expensive.
        # Path around (cost 1s) is longer in steps but cheaper.
        grid = [
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is not None
        
        # Path via middle: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) cost: 1+1+1+1+1 = 5? 
        # Wait, let's construct a clearer example.
        # S 1 1
        # 1 100 1
        # 1 1 1 E
        # Path 1 (Top/Right): (0,0)->(0,1)->(0,2)->(1,2)->(2,2) Cost: 1+1+1+1+1 = 5
        # Path 2 (Bottom/Left): (0,0)->(1,0)->(2,0)->(2,1)->(2,2) Cost: 1+1+1+1+1 = 5
        # Let's make the direct path expensive.
        
        grid = [
            [1, 1, 1],
            [1, 100, 1],
            [1, 1, 1]
        ]
        # Actually, let's force a choice:
        # S 1 1
        # 1 10 1
        # 1 1 1 E
        # If we go (0,0)->(1,0)->(2,0)->(2,1)->(2,2) cost = 1+1+1+1+1 = 5
        # If we go (0,0)->(0,1)->(0,2)->(1,2)->(2,2) cost = 1+1+1+1+1 = 5
        # Let's make the "short" path go through a high cost.
        
        grid = [
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ]
        # Start (0,0), End (2,2).
        # Option A: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) Cost: 1+1+1+1+1 = 5
        # Option B: (0,0)->(1,0)->(2,0)->(2,1)->(2,2) Cost: 1+1+1+1+1 = 5
        # Option C: (0,0)->(1,0)->(1,1)->(1,2)->(2,2) Cost: 1+10+1+1+1 = 14 (Bad)
        
        # Let's try a specific case where the heuristic might mislead if not weighted correctly.
        # S 1 1
        # 1 10 1
        # 1 1 1 E
        # Actually, let's just ensure it picks the low cost path if one exists.
        grid = [
            [1, 1, 1],
            [1, 10, 1],
            [1, 1, 1]
        ]
        # Let's change the grid to force a specific optimal path.
        # S 1 1
        # 1 10 1
        # 1 1 1 E
        # If we make the top path expensive?
        grid = [
            [1, 10, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        # Start (0,0) to (2,2).
        # Path 1 (Top): (0,0)->(0,1)->(0,2)->(1,2)->(2,2) Cost: 1+10+1+1+1 = 14
        # Path 2 (Bottom): (0,0)->(1,0)->(2,0)->(2,1)->(2,2) Cost: 1+1+1+1+1 = 5
        # A* should pick Path 2.
        
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is not None
        total_cost = calculate_path_cost(grid, path)
        assert total_cost == 5, f"Expected cost 5, got {total_cost}. Path: {path}"
        
        # Verify it didn't take the expensive route
        assert (0, 1) not in path

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
        assert calculate_path_cost(grid, path) == grid[1][1]

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
            
        # Test negative index
        with pytest.raises(ValueError):
            astar.find_path((0, -1), (0, 0))

    def test_start_or_end_is_wall(self):
        """Additional edge case: Start or End is a wall (0)."""
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

*   **Heuristic**: Uses Manhattan distance (`|x1-x2| + |y1-y2|`), which is admissible for 4-directional movement.
*   **Optimality**: The algorithm uses a priority queue (min-heap) ordered by `f_score = g_score + heuristic`. It updates `g_score` (actual cost from start) whenever a cheaper path to a node is found, ensuring the final path is optimal even with weighted cells.
*   **Data Structures**:
    *   `heapq`: Used for the open set to efficiently retrieve the node with the lowest `f_score`.
    *   `dict`: Used for `g_score`, `f_score`, and `came_from` for O(1) lookups.
    *   `set`: Used for `open_set_hash` to quickly check if a node is currently in the priority queue.
*   **Edge Cases**: Explicitly handles start==end, out-of-bounds (raises `ValueError`), and start/end being walls (returns `None`).
