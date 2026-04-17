Here's the complete implementation with type hints, docstrings, and the requested pytest tests:

```python
import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """A* pathfinding implementation on a weighted 2D grid."""

    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the grid with movement costs. 0 represents an impassable wall."""
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two coordinates."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find the shortest path from start to end using A* algorithm.

        Args:
            start: Starting (row, col) coordinate.
            end: Ending (row, col) coordinate.

        Returns:
            List of coordinates representing the optimal path, or None if no path exists.
        """
        # Edge case: out of bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start coordinates out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End coordinates out of bounds")

        # Edge case: walls
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Edge case: start equals end
        if start == end:
            return [start]

        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))
        came_from = {}
        g_score = {start: 0}
        closed_set = set()
        counter = 1

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

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

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                nr, nc = neighbor

                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr][nc] == 0:
                    continue
                if neighbor in closed_set:
                    continue

                tentative_g = g_score[current] + self.grid[nr][nc]

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f, counter, neighbor))
                    counter += 1

        return None
```

### Pytest Tests

```python
import pytest

def _path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Helper to calculate total movement cost of a path."""
    return sum(grid[r][c] for r, c in path)

def test_simple_uniform_path():
    """Test simple path on a uniform grid where all cells cost 1."""
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    
    # Verify adjacency
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1
        
    # Optimal cost: 5 cells * 1 cost = 5
    assert _path_cost(grid, path) == 5

def test_path_around_obstacles():
    """Test pathfinding around a single wall obstacle."""
    grid = [[1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert (1, 1) not in path  # Must avoid the wall
    
    # Optimal cost: 5 cells * 1 cost = 5
    assert _path_cost(grid, path) == 5

def test_weighted_grid():
    """Test that A* prefers lower-cost cells over shorter geometric paths."""
    grid = [[1, 10, 1],
            [1, 10, 1],
            [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is not None
    # Direct path cost: 1 + 10 + 1 = 12
    # Optimal path goes around the high-cost column: 7 cells * 1 = 7
    assert _path_cost(grid, path) == 7

def test_no_path_exists():
    """Test when the destination is completely blocked by walls."""
    grid = [[1, 1, 1],
            [0, 0, 0],
            [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is None

def test_start_equals_end():
    """Test when start and end coordinates are identical."""
    grid = [[1, 1],
            [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    
    assert path == [(1, 1)]

def test_invalid_coordinates():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1],
            [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
        
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))
```

### Key Implementation Details:
1. **Admissibility**: Manhattan distance is admissible for 4-directional movement when all cell costs are ≥ 1, guaranteeing optimality.
2. **Heap Management**: Uses a counter to break ties deterministically in the min-heap, preventing comparison errors between tuples.
3. **Closed Set**: Prevents reprocessing finalized nodes and handles stale heap entries efficiently.
4. **Cost Calculation**: `g_score` tracks cumulative cost from start. The path cost includes all visited cells (start to end inclusive), matching standard pathfinding conventions.
5. **Edge Cases**: Explicitly handled before algorithm initialization to avoid unnecessary computation and ensure correct return types/exceptions.