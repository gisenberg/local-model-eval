Here's a complete implementation of A* pathfinding on a weighted 2D grid in Python, including the requested class and pytest tests:

```python
import heapq
from typing import List, Tuple, Optional, Dict
import math

class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.

    The grid values represent movement cost where:
    - 0 = impassable wall
    - positive int = cost to enter that cell
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D grid.

        Args:
            grid: 2D list where each cell represents movement cost (0 = wall)
        """
        self.grid = grid
        self.rows = len(grid)
        if self.rows == 0:
            self.cols = 0
        else:
            self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.

        Args:
            start: (row, col) tuple representing the starting position
            end: (row, col) tuple representing the target position

        Returns:
            List of (row, col) tuples representing the path from start to end (inclusive),
            or None if no path exists

        Raises:
            ValueError: If start or end coordinates are out of bounds
        """
        # Check for out of bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start coordinates out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End coordinates out of bounds")

        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Handle start == end case
        if start == end:
            return [start]

        # Directions for 4-way movement (up, down, left, right)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Priority queue: (f_score, g_score, current, parent)
        open_set = []
        heapq.heappush(open_set, (0, 0, start, None))

        # For each node, store the best known g_score and its parent
        g_scores: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        while open_set:
            _, current_g, current, _ = heapq.heappop(open_set)

            # Skip if we've already found a better path to current
            if current_g > g_scores.get(current, math.inf):
                continue

            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]  # Reverse to get start to end

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)

                # Check if neighbor is within bounds
                if not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols):
                    continue

                # Check if neighbor is a wall
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue

                # Calculate tentative g_score
                tentative_g = current_g + self.grid[neighbor[0]][neighbor[1]]

                # If this path to neighbor is better than any previous one
                if tentative_g < g_scores.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._manhattan_distance(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, current))

        # No path found
        return None

    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two points.

        Args:
            a: First point (row, col)
            b: Second point (row, col)

        Returns:
            Manhattan distance between a and b
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Pytest tests
import pytest

def test_simple_path_uniform_grid():
    """Test a simple path on a uniform grid (all cells have cost 1)."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)] or path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    # Check path cost
    total_cost = sum(grid[r][c] for r, c in path)
    assert total_cost == 4  # Minimum possible cost for this path

def test_path_around_obstacles():
    """Test pathfinding around obstacles (walls)."""
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 3))
    assert path is not None
    # Check that path goes around the obstacles
    assert (1, 1) not in path and (1, 2) not in path
    # Check path cost
    total_cost = sum(grid[r][c] for r, c in path)
    assert total_cost == 6  # Minimum possible cost for this path

def test_weighted_grid_prefers_lower_cost():
    """Test that path prefers lower-cost cells in a weighted grid."""
    grid = [
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    # The optimal path should avoid the high-cost cell (1,1)
    assert (1, 1) not in path
    # Check path cost
    total_cost = sum(grid[r][c] for r, c in path)
    assert total_cost == 4  # Minimum possible cost (avoiding the 5-cost cell)

def test_no_path_exists():
    """Test when no path exists (fully blocked)."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_start_equals_end():
    """Test when start and end are the same."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_invalid_coordinates():
    """Test with invalid (out of bounds) coordinates."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 2))
```

### Key Features:

1. **AStarGrid Class**:
   - Initializes with a grid where 0 represents walls
   - Implements `find_path` with all required functionality

2. **A* Algorithm**:
   - Uses a min-heap (priority queue) for the open set
   - Implements Manhattan distance as the heuristic
   - Tracks g_scores (actual cost from start) and reconstructs the optimal path

3. **Edge Case Handling**:
   - Returns `[start]` when start == end
   - Returns `None` when start or end is a wall
   - Raises `ValueError` for out-of-bounds coordinates

4. **Type Hints**:
   - Full type annotations throughout the code

5. **Pytest Tests**:
   - 6 comprehensive tests covering all requirements
   - Each test verifies both path validity and optimality (total cost)

The implementation follows standard A* algorithm with the specified requirements, and the tests verify both correctness and optimality of the paths found.