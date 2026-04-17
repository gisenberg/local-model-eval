Here's a complete, robust implementation of A* pathfinding on a weighted 2D grid, meeting all your requirements.

---

### ✅ Implementation (`astar_grid.py`)

```python
import heapq
from typing import List, Tuple, Optional
import math

class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.
    
    Grid values:
        - 0: impassable wall
        - positive int: cost to enter that cell
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid.
        
        :param grid: 2D list of non-negative integers (0 = wall, >0 = movement cost)
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        # Validate grid dimensions are consistent
        for row in grid:
            if len(row) != self.cols:
                raise ValueError("All rows must have the same length")

    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_passable(self, pos: Tuple[int, int]) -> bool:
        """Check if cell at pos is passable (non-zero cost)."""
        r, c = pos
        return self.grid[r][c] > 0

    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return valid 4-directional neighbors."""
        r, c = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self._in_bounds((nr, nc)) and self._is_passable((nr, nc)):
                neighbors.append((nr, nc))
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the minimum-cost path from start to end using A*.
        
        :param start: (row, col) starting position
        :param end: (row, col) ending position
        :return: list of positions from start to end inclusive, or None if no path exists
        :raises ValueError: if start or end is out of bounds
        """
        # Validate bounds
        if not self._in_bounds(start):
            raise ValueError(f"Start position {start} is out of bounds")
        if not self._in_bounds(end):
            raise ValueError(f"End position {end} is out of bounds")
        
        # Handle same start/end
        if start == end:
            if self._is_passable(start):
                return [start]
            else:
                return None  # start/end is a wall
        
        # Early exit: start or end is a wall
        if not self._is_passable(start) or not self._is_passable(end):
            return None

        # A* algorithm
        # g_score[(r,c)] = cost from start to (r,c)
        g_score = {start: 0}
        # f_score[(r,c)] = g_score + heuristic
        f_score = {start: self._manhattan_distance(start, end)}
        
        # open_set: min-heap of (f_score, counter, pos)
        # counter breaks ties deterministically
        counter = 0
        open_set = [(f_score[start], counter, start)]
        heapq.heapify(open_set)
        
        # came_from: for path reconstruction
        came_from = {}
        
        # closed set: positions already evaluated
        closed_set = set()
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]  # reverse to get start -> end
            
            if current in closed_set:
                continue
            closed_set.add(current)
            
            for neighbor in self._neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Cost to move to neighbor = grid value at neighbor
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._manhattan_distance(neighbor, end)
                    
                    # Update heap: push new (f, counter, neighbor)
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
        
        # No path found
        return None
```

---

### ✅ Tests (`test_astar_grid.py`)

```python
import pytest
from astar_grid import AStarGrid
from typing import List, Tuple

def path_cost(grid: AStarGrid, path: List[Tuple[int, int]]) -> int:
    """Compute total cost of a path (sum of entry costs, excluding start cell cost)."""
    if len(path) <= 1:
        return 0
    total = 0
    for pos in path[1:]:  # skip start
        r, c = pos
        total += grid.grid[r][c]
    return total

def test_simple_path_uniform_grid():
    grid_data = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid_data)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Manhattan distance = 4 steps → 4 moves → cost = 4 (each step costs 1)
    assert len(path) == 5  # 5 positions for 4 moves
    assert path_cost(astar, path) == 4

def test_path_around_obstacles():
    grid_data = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]
    astar = AStarGrid(grid_data)
    path = astar.find_path((0, 0), (2, 4))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 4)
    # Path must go around: e.g., (0,0)→(0,1)→(0,2)→(0,3)→(0,4)→(1,4)→(2,4)
    # But (0,2) is blocked? Wait: row 1 has walls at cols 1-3.
    # So path: (0,0)→(0,1)→(0,2)→(0,3)→(0,4)→(1,4)→(2,4) is invalid because (0,2) is passable? 
    # Actually grid[1][1:4] = [0,0,0] → walls at (1,1), (1,2), (1,3)
    # So path: (0,0)→(1,0)→(2,0)→(2,1)→(2,2)→(2,3)→(2,4) → cost = 6
    assert path_cost(astar, path) == 6

def test_weighted_grid_prefers_lower_cost():
    grid_data = [
        [1, 100, 1],
        [1, 100, 1],
        [1, 100, 1]
    ]
    astar = AStarGrid(grid_data)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Should go around the high-cost column (col=1)
    # Path: (0,0)→(1,0)→(2,0)→(2,1)→(2,2) → cost = 1+1+1+1 = 4
    # vs (0,0)→(0,1)→(0,2)→(1,2)→(2,2) → cost = 100+1+100+1 = 202
    assert path_cost(astar, path) == 4
    # Verify path avoids high-cost cells
    for r, c in path:
        assert grid_data[r][c] == 1, f"Path includes high-cost cell ({r},{c})"

def test_no_path_exists():
    grid_data = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid_data)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_start_equals_end():
    grid_data = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid_data)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]
    
    # Also test wall case
    grid_data_wall = [
        [0, 1],
        [1, 1]
    ]
    astar_wall = AStarGrid(grid_data_wall)
    path_wall = astar_wall.find_path((0, 0), (0, 0))
    assert path_wall is None

def test_invalid_coordinates():
    grid_data = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid_data)
    
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 1))
    
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 2))
```

---

### ✅ How to Run Tests

1. Save the implementation as `astar_grid.py`
2. Save the tests as `test_astar_grid.py`
3. Install pytest if needed: `pip install pytest`
4. Run: `pytest test_astar_grid.py -v`

---

### ✅ Notes on Correctness & Edge Cases

- **Optimality**: Uses standard A* with consistent heuristic (Manhattan), so optimal.
- **Wall handling**: `0` cost = wall → impassable.
- **Path cost**: Only includes cost of *entering* cells (so start cell cost is not counted — matches typical grid pathfinding semantics).
- **Tie-breaking**: Uses a counter to avoid comparing tuples with non-comparable types in heap.
- **Bounds checking**: Validates both start and end before processing.
- **Same start/end**: Returns `[start]` only if passable.

Let me know if you'd like a visualization helper or performance benchmarks!