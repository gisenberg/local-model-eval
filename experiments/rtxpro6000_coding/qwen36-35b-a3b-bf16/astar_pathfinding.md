Here's the complete implementation with type hints, docstrings, and the requested pytest tests.

### `astar.py`
```python
import heapq
from typing import List, Tuple, Optional, Dict, Set

class AStarGrid:
    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the grid with movement costs. 0 = wall, positive int = entry cost."""
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two coordinates."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find the shortest path from start to end using A* algorithm."""
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start coordinates out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End coordinates out of bounds")

        # Check for walls
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Start equals end
        if start == end:
            return [start]

        # A* initialization
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))
        counter = 1

        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set: Set[Tuple[int, int]] = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            if current in closed_set:
                continue
            closed_set.add(current)

            r, c = current
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.grid[nr][nc] == 0:
                        continue

                    neighbor = (nr, nc)
                    tentative_g = g_score[current] + self.grid[nr][nc]

                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + self._heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score, counter, neighbor))
                        counter += 1

        return None
```

### `test_astar.py`
```python
import pytest
from astar import AStarGrid

def calculate_path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Calculate total movement cost for a path (excludes start cell cost)."""
    return sum(grid[r][c] for r, c in path[1:])

def test_simple_path_uniform_grid():
    """Test 1: Simple path on a uniform grid where all cells cost 1."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Shortest path in 3x3 uniform grid requires 4 steps
    assert calculate_path_cost(grid, path) == 4

def test_path_around_obstacles():
    """Test 2: Pathfinding around a single wall obstacle."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert (1, 1) not in path  # Should avoid the wall
    assert calculate_path_cost(grid, path) == 4

def test_weighted_grid():
    """Test 3: Weighted grid where path prefers lower-cost cells over shorter distance."""
    grid = [
        [1, 10, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is not None
    # Direct path cost: 10 + 1 = 11
    # Down route cost: 1+1+1+1+1+1 = 6
    assert calculate_path_cost(grid, path) == 6
    assert (0, 1) not in path  # Should avoid the expensive cell

def test_no_path_exists():
    """Test 4: Fully blocked grid where no path exists."""
    grid = [
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is None

def test_start_equals_end():
    """Test 5: Start and end coordinates are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    
    assert path == [(1, 1)]
    assert calculate_path_cost(grid, path) == 0

def test_invalid_coordinates():
    """Test 6: Coordinates outside grid boundaries raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, -1), (0, 0))
```

### Key Implementation Details:
1. **Admissibility & Optimality**: Manhattan distance is admissible for 4-directional movement when all cell costs are ≥ 1, guaranteeing the returned path has the minimum total cost.
2. **Heap Tie-Breaking**: A monotonic counter is used in the heap tuple `(f_score, counter, position)` to prevent `heapq` from comparing coordinate tuples when `f_score` values are equal.
3. **Cost Calculation**: Path cost sums the entry cost of each cell *after* the start cell, matching standard grid pathfinding conventions.
4. **Edge Cases**: Explicitly handled before algorithm initialization to avoid unnecessary computation and ensure correct return types/exceptions.
5. **Type Hints**: Full static typing throughout using `typing` module for maximum compatibility and IDE support.