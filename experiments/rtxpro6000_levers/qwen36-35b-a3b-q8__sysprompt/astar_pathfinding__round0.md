Here is the complete, production-ready implementation followed by the requested pytest suite.

### Implementation

```python
import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """A* pathfinding algorithm on a weighted 2D grid."""

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid with movement costs.
        0 represents an impassable wall, positive integers represent the cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """Calculate Manhattan distance between two coordinates."""
        return abs(r1 - r2) + abs(c1 - c2)

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end.
        Returns a list of (row, col) coordinates from start to end inclusive,
        or None if no path exists.
        """
        sr, sc = start
        er, ec = end

        # Validate bounds
        if not (0 <= sr < self.rows and 0 <= sc < self.cols):
            raise ValueError(f"Start coordinates {start} are out of bounds.")
        if not (0 <= er < self.rows and 0 <= ec < self.cols):
            raise ValueError(f"End coordinates {end} are out of bounds.")

        # Check for walls
        if self.grid[sr][sc] == 0 or self.grid[er][ec] == 0:
            return None

        # Start equals end
        if start == end:
            return [start]

        # A* initialization
        open_set: List[Tuple[float, int, int, int]] = []
        heapq.heappush(open_set, (0.0, 0, sr, sc))  # (f_score, counter, row, col)
        counter = 1

        g_score: Dict[Tuple[int, int], float] = {(sr, sc): 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_set:
            _, _, curr_r, curr_c = heapq.heappop(open_set)

            if (curr_r, curr_c) == end:
                # Reconstruct path
                path: List[Tuple[int, int]] = []
                curr = end
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.append(start)
                return path[::-1]

            # Explore neighbors (up, down, left, right)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = curr_r + dr, curr_c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != 0:
                    tentative_g = g_score[(curr_r, curr_c)] + self.grid[nr][nc]

                    if tentative_g < g_score.get((nr, nc), float('inf')):
                        came_from[(nr, nc)] = (curr_r, curr_c)
                        g_score[(nr, nc)] = tentative_g
                        f = tentative_g + self._heuristic(nr, nc, er, ec)
                        heapq.heappush(open_set, (f, counter, nr, nc))
                        counter += 1

        return None
```

### Pytest Suite

```python
import pytest
from typing import List, Tuple

def _calculate_path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Helper to calculate total movement cost of a path (excludes start cell cost)."""
    return sum(grid[r][c] for r, c in path[1:])

def test_simple_path_uniform_grid():
    """Simple path on a uniform grid where all cells cost 1."""
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    
    # Verify path validity: each step must be adjacent (no diagonals)
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1
        
    # Optimality: shortest path on uniform grid is 4 steps -> cost 4
    assert _calculate_path_cost(grid, path) == 4

def test_path_around_obstacles():
    """Path must navigate around a wall (0 cost cell)."""
    grid = [[1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert _calculate_path_cost(grid, path) == 4
    assert (1, 1) not in path  # Must avoid the wall

def test_weighted_grid_prefers_lower_cost():
    """A* must prefer lower-cost cells even if Manhattan distance is equal."""
    grid = [[1, 10, 1],
            [1, 10, 1],
            [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    # Optimal path avoids the 10s: (0,0)->(1,0)->(2,0)->(2,1)->(2,2) cost = 4
    # Alternative path via top/right costs 13
    assert _calculate_path_cost(grid, path) == 4

def test_no_path_exists():
    """Return None when start and end are separated by impassable walls."""
    grid = [[1, 1, 1],
            [1, 0, 1],
            [1, 0, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is None

def test_start_equals_end():
    """Return single-element list when start and end coordinates match."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    
    assert path == [(1, 1)]
    assert _calculate_path_cost(grid, path) == 0

def test_invalid_coordinates():
    """Raise ValueError when start or end coordinates are out of grid bounds."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 0))
```

### Key Design Decisions & Invariants Preserved
1. **Optimality Guarantee**: Manhattan distance is an *admissible* and *consistent* heuristic for 4-directional movement with positive edge weights. This ensures A* expands nodes in order of true optimal cost, guaranteeing the returned path has the minimum total cost.
2. **Cost Semantics**: The prompt specifies costs are for *entering* a cell. Therefore, `g_score[start] = 0` and each step adds `grid[nr][nc]`. Path cost is the sum of all cells except the start.
3. **Heap Tie-Breaking**: A `counter` is included in the heap tuple `(f, counter, r, c)` to prevent `heapq` from comparing coordinate tuples when `f` scores are equal, avoiding `TypeError` and ensuring deterministic behavior.
4. **Edge Case Ordering**: Bounds are checked first (raises `ValueError`), then wall status (returns `None`), then `start == end` (returns `[start]`). This matches the exact contract specified.
5. **Memory Efficiency**: `f_score` is not stored in a dictionary since it's deterministically `g_score + heuristic`. Only `g_score` and `came_from` are cached, reducing memory overhead during search.