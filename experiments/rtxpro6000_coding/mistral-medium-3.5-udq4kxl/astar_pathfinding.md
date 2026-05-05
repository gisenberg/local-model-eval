Here's a complete implementation of A* pathfinding on a weighted 2D grid in Python, including the requested pytest tests:

```python
import heapq
from typing import List, Tuple, Optional, Dict
import math

class AStarGrid:
    """
    A* pathfinding implementation for a weighted 2D grid.
    Grid values represent movement cost (0 = impassable wall, positive int = cost to enter that cell).
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid for pathfinding.

        Args:
            grid: 2D list where each cell's value represents movement cost.
                  0 means impassable, positive integers are movement costs.
        """
        self.grid = grid
        self.rows = len(grid)
        if self.rows == 0:
            self.cols = 0
        else:
            self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm with Manhattan distance heuristic.

        Args:
            start: (row, col) tuple representing the starting position
            end: (row, col) tuple representing the target position

        Returns:
            List of (row, col) tuples representing the path from start to end (inclusive),
            or None if no path exists.

        Raises:
            ValueError: If start or end coordinates are out of bounds.
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

        # Priority queue: (f_score, g_score, current_position)
        open_set = []
        heapq.heappush(open_set, (0, 0, start))

        # For each node, store the best known g_score and the node that leads to it
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}

        # f_score = g_score + heuristic
        f_score: Dict[Tuple[int, int], float] = {start: self._manhattan_distance(start, end)}

        open_set_hash = {start}  # For quick lookup

        while open_set:
            current = heapq.heappop(open_set)[2]
            open_set_hash.remove(current)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)

                # Check if neighbor is within bounds
                if not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols):
                    continue

                # Check if neighbor is a wall
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue

                # Calculate tentative g_score
                movement_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g_score = g_score[current] + movement_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._manhattan_distance(neighbor, end)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))
                        open_set_hash.add(neighbor)

        return None  # No path found

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

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from start to current using the came_from dictionary.

        Args:
            came_from: Dictionary mapping each node to its predecessor
            current: The current node (end of path)

        Returns:
            List of nodes representing the path from start to current
        """
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

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
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Check path length (should be 5 for this simple case)
    assert len(path) == 5
    # Check total cost (should be 4 since we move 4 steps)
    total_cost = sum(grid[r][c] for r, c in path[1:])  # exclude start
    assert total_cost == 4

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
    assert path[0] == (0, 0)
    assert path[-1] == (2, 3)
    # Check that path goes around the obstacles
    assert (1, 1) not in path
    assert (1, 2) not in path
    # Check total cost (should be 6)
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 6

def test_weighted_grid_prefers_lower_cost():
    """Test that path prefers lower-cost cells in a weighted grid."""
    grid = [
        [1, 1, 1, 1],
        [1, 5, 5, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 3))
    assert path is not None
    # The optimal path should go around the high-cost cells
    assert (1, 1) not in path
    assert (1, 2) not in path
    # Check total cost (should be 6, going around the 5-cost cells)
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 6

def test_no_path_exists():
    """Test case where no path exists (fully blocked)."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_start_equals_end():
    """Test case where start equals end."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_invalid_coordinates():
    """Test case with invalid coordinates (out of bounds)."""
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

if __name__ == "__main__":
    pytest.main([__file__])
```

### Key Features:

1. **AStarGrid Class**:
   - Initializes with a grid where 0 represents walls and positive integers represent movement costs
   - Implements the A* algorithm with Manhattan distance heuristic

2. **find_path Method**:
   - Handles all edge cases (start == end, walls, out of bounds)
   - Uses a min-heap (priority queue) for the open set
   - Returns the optimal path or None if no path exists

3. **Helper Methods**:
   - `_manhattan_distance`: Calculates the heuristic
   - `_reconstruct_path`: Builds the path from the came_from dictionary

4. **Pytest Tests**:
   - `test_simple_path_uniform_grid`: Tests basic pathfinding on a uniform grid
   - `test_path_around_obstacles`: Tests pathfinding around walls
   - `test_weighted_grid_prefers_lower_cost`: Verifies the algorithm prefers lower-cost paths
   - `test_no_path_exists`: Tests when no path is possible
   - `test_start_equals_end`: Tests the edge case where start and end are the same
   - `test_invalid_coordinates`: Tests out-of-bounds handling

The implementation ensures the path is optimal (minimum total cost) by properly maintaining the g_score and f_score dictionaries and using the priority queue to always expand the most promising node first.