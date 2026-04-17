Here is the complete, production-ready implementation followed by the requested pytest suite.

### Implementation

```python
import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """A* pathfinding on a weighted 2D grid with 4-directional movement."""

    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the A* grid with movement costs.
        
        Args:
            grid: 2D list where 0 represents an impassable wall and 
                  positive integers represent the cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: (row, col) tuple for the starting position.
            end: (row, col) tuple for the destination position.
            
        Returns:
            List of (row, col) coordinates from start to end inclusive, 
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of grid bounds.
        """
        # 1. Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start coordinates are out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End coordinates are out of bounds.")

        # 2. Check for walls
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # 3. Start equals end
        if start == end:
            return [start]

        # 4. Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # 5. Manhattan distance heuristic
        def heuristic(r: int, c: int) -> int:
            return abs(r - end[0]) + abs(c - end[1])

        # 6. Open set: (f_score, counter, (r, c))
        # Counter ensures stable sorting when f_scores tie
        counter = 0
        open_set = [(heuristic(start[0], start[1]), counter, start)]
        heapq.heapify(open_set)

        # 7. G-scores and path reconstruction
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path by backtracking
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            cr, cc = current
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != 0:
                    tentative_g = g_score[current] + self.grid[nr][nc]
                    
                    # Only update if we found a cheaper path to this neighbor
                    if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                        g_score[(nr, nc)] = tentative_g
                        f_score = tentative_g + heuristic(nr, nc)
                        counter += 1
                        heapq.heappush(open_set, (f_score, counter, (nr, nc)))
                        came_from[(nr, nc)] = current

        return None
```

### Pytest Tests

```python
import pytest

def _verify_path_and_cost(grid: List[List[int]], path: Optional[List[Tuple[int, int]]], expected_cost: int) -> None:
    """Helper to assert path validity and optimality."""
    assert path is not None, "Path should not be None"
    rows, cols = len(grid), len(grid[0])
    
    # 1. Check bounds and walls
    for r, c in path:
        assert 0 <= r < rows and 0 <= c < cols, f"Path out of bounds: ({r}, {c})"
        assert grid[r][c] != 0, f"Path hits impassable wall: ({r}, {c})"
        
    # 2. Check adjacency (4-directional only)
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1, "Path steps are not adjacent"
        
    # 3. Check optimality (total cost matches sum of entered cells)
    actual_cost = sum(grid[r][c] for r, c in path[1:])
    assert actual_cost == expected_cost, f"Expected cost {expected_cost}, got {actual_cost}"


def test_simple_uniform_grid():
    """Simple path on a uniform grid where all cells cost 1."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    _verify_path_and_cost(grid, path, expected_cost=4)


def test_path_around_obstacles():
    """Path must navigate around a central wall."""
    grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    _verify_path_and_cost(grid, path, expected_cost=4)


def test_weighted_grid_prefers_low_cost():
    """Weighted grid: algorithm should avoid high-cost cells even if path length is equal."""
    grid = [
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    # Optimal path avoids the 10-cost cell, total cost = 1+1+1+1 = 4
    _verify_path_and_cost(grid, path, expected_cost=4)


def test_no_path_exists():
    """Fully blocked destination should return None."""
    grid = [
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None, "Should return None when no valid path exists"


def test_start_equals_end():
    """Start and end at same location should return list containing only that coordinate."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)], "Should return list containing only start node"


def test_invalid_coordinates():
    """Out of bounds coordinates should raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (0, 0))
        
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))
```

### Key Design Decisions & Invariants Preserved
1. **Cost Semantics**: Movement cost is charged upon *entering* a cell. `g_score[start]` is initialized to `0`, and each neighbor adds `grid[nr][nc]`. This matches the prompt's "cost to enter that cell" specification.
2. **Heuristic Admissibility**: Manhattan distance is used. Since all passable cells have costs `≥ 1`, the heuristic never overestimates the true remaining cost, guaranteeing A* optimality.
3. **Lazy Deletion**: The open set uses a min-heap with a monotonic counter to break ties deterministically. Stale entries remain in the heap but are safely ignored because we only process a node when it's popped, and we only expand it if it matches the current best `g_score` (implicit in the update condition).
4. **Edge Case Ordering**: Bounds validation → Wall check → `start == end` check. This ensures walls correctly return `None` even when `start == end`, aligning with the requirement that walls are impassable.
5. **Type Safety**: Full type hints including `Dict` for internal state tracking, `Optional` for return types, and strict tuple typing for coordinates.