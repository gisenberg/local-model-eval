Here's the complete implementation with type hints, docstrings, and the requested pytest tests.

### Implementation

```python
import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """A* pathfinding implementation on a weighted 2D grid."""

    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the grid with movement costs.
        
        Args:
            grid: 2D list where 0 represents an impassable wall,
                  and positive integers represent the cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """Calculate Manhattan distance between two coordinates.
        
        Args:
            r1, c1: Row and column of the first coordinate.
            r2, c2: Row and column of the second coordinate.
            
        Returns:
            Manhattan distance as an integer.
        """
        return abs(r1 - r2) + abs(c1 - c2)

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Tuple (row, col) of the starting position.
            end: Tuple (row, col) of the target position.
            
        Returns:
            List of (row, col) coordinates from start to end inclusive,
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of grid bounds.
        """
        sr, sc = start
        er, ec = end

        # Validate bounds
        if not (0 <= sr < self.rows and 0 <= sc < self.cols):
            raise ValueError(f"Start coordinate {start} is out of grid bounds.")
        if not (0 <= er < self.rows and 0 <= ec < self.cols):
            raise ValueError(f"End coordinate {end} is out of grid bounds.")

        # Check for walls
        if self.grid[sr][sc] == 0 or self.grid[er][ec] == 0:
            return None

        # Handle start == end
        if start == end:
            return [start]

        # A* initialization
        g_scores = {start: 0}
        f_scores = {start: self._heuristic(sr, sc, er, ec)}
        open_heap = [(f_scores[start], 0, start)]
        came_from = {start: None}

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_heap:
            f, g, current = heapq.heappop(open_heap)
            cr, cc = current

            # If we reached the end, reconstruct and return path
            if current == end:
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = came_from.get(node)
                return path[::-1]

            # Skip if we found a better path to this node already
            if g > g_scores.get(current, float('inf')):
                continue

            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                neighbor = (nr, nc)

                # Check bounds and walls
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] > 0:
                    tentative_g = g + self.grid[nr][nc]

                    if tentative_g < g_scores.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_scores[neighbor] = tentative_g
                        f_scores[neighbor] = tentative_g + self._heuristic(nr, nc, er, ec)
                        heapq.heappush(open_heap, (f_scores[neighbor], tentative_g, neighbor))

        return None
```

### Pytest Tests

```python
import pytest
from typing import List, Tuple

def _path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Calculate total movement cost for a path (excluding start cell cost)."""
    return sum(grid[r][c] for r, c in path[1:])

def test_simple_uniform_path():
    """Test 1: Simple path on a uniform grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (2, 2)
    
    # Validate adjacency (4-directional movement)
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1
        
    # Optimal cost on 3x3 uniform grid is 4 steps
    assert _path_cost(grid, path) == 4

def test_path_around_obstacles():
    """Test 2: Path around obstacles."""
    grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    # Must navigate around the center wall: 6 steps
    assert _path_cost(grid, path) == 6

def test_weighted_grid():
    """Test 3: Weighted grid prefers lower-cost cells."""
    grid = [[1, 10, 1], [1, 10, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is not None
    # Avoids the 10-cost column, routes around the bottom
    # Path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2)
    assert _path_cost(grid, path) == 6

def test_no_path_exists():
    """Test 4: No path exists (fully blocked)."""
    grid = [[1, 1, 1], [0, 0, 0], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is None

def test_start_equals_end():
    """Test 5: Start equals end."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    
    assert path == [(1, 1)]
    assert _path_cost(grid, path) == 0

def test_invalid_coordinates():
    """Test 6: Invalid coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 5))
```

### Key Design Notes:
1. **Optimality Guarantee**: Manhattan distance is an *admissible* and *consistent* heuristic for 4-directional movement on grids with positive costs. This guarantees A* finds the minimum-cost path without needing to re-evaluate nodes.
2. **Heap Management**: The open set uses `heapq` storing `(f_score, g_score, coordinate)`. Python's tuple comparison ensures deterministic ordering when scores tie.
3. **Stale Entry Handling**: The `if g > g_scores.get(current, float('inf')): continue` check efficiently skips outdated heap entries, a standard optimization in A* implementations.
4. **Cost Calculation**: The start cell's cost is not included in the total path cost, matching standard grid-based pathfinding conventions where cost represents the expense to *enter* a cell.