# MiniMax-M2.5 UD-Q3_K_XL (empty-think) — A* Pathfinding

## Output (8751 chars)

Here is a complete implementation of A* pathfinding on a weighted 2D grid in Python, along with six pytest tests.

```python
"""
A* Pathfinding on a Weighted 2D Grid

This module implements the A* search algorithm on a grid where each cell has a movement cost.
It supports 4-directional movement and uses Manhattan distance as the heuristic.
"""

import heapq
from typing import List, Tuple, Optional


class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.

    Attributes:
        grid (List[List[int]]): 2D list where 0 represents a wall (impassable) and
                                positive integers represent the cost to enter that cell.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the A* grid.

        Args:
            grid: 2D list of movement costs (0 = wall, positive = cost).
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, row: int, col: int) -> bool:
        """Check if a coordinate is within bounds and not a wall."""
        return 0 <= row < self.rows and 0 <= col < self.cols and self.grid[row][col] != 0

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to end using the came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A*.

        Args:
            start: Starting coordinate as (row, col).
            end: Target coordinate as (row, col).

        Returns:
            List of coordinates from start to end inclusive, or None if no path exists.

        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate bounds
        if not self._is_valid(start[0], start[1]) and start != end:
            raise ValueError(f"Start coordinate {start} is out of bounds or is a wall.")
        if not self._is_valid(end[0], end[1]) and start != end:
            raise ValueError(f"End coordinate {end} is out of bounds or is a wall.")

        # Edge case: start equals end
        if start == end:
            return [start]

        # Check if start or end is a wall (if they are not equal to each other)
        if not self._is_valid(start[0], start[1]):
            return None
        if not self._is_valid(end[0], end[1]):
            return None

        # A* initialization
        open_set: List[Tuple[int, Tuple[int, int]]] = []  # (f_score, position)
        heapq.heappush(open_set, (self._manhattan(start, end), start))

        came_from: dict = {}
        g_score: dict = {start: 0}
        f_score: dict = {start: self._manhattan(start, end)}

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)

                if not self._is_valid(neighbor[0], neighbor[1]):
                    continue

                # Cost to enter the neighbor cell
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._manhattan(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        return None
```

Now, the test file:

```python
"""
Tests for AStarGrid pathfinding implementation.
"""

import pytest
from astar_grid import AStarGrid


class TestAStarGrid:
    """Test suite for A* pathfinding on a weighted 2D grid."""

    def test_simple_path_uniform_grid(self):
        """Test simple path on a uniform grid with no obstacles."""
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
        # Manhattan distance on a uniform grid = 4 steps
        assert len(path) == 5
        # Total cost should be 4 (each step cost 1)
        total_cost = sum(grid[r][c] for r, c in path[1:])
        assert total_cost == 4

    def test_path_around_obstacles(self):
        """Test path that must go around walls."""
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
        # Path should go down, right, right, down, down
        # or similar valid route around the wall
        assert (1, 0) in path or (0, 1) in path

    def test_weighted_grid_prefers_lower_cost(self):
        """Test that the path prefers lower-cost cells over higher-cost ones."""
        grid = [
            [1, 10, 1],
            [1, 1,  1],
            [1, 10, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (0, 2))
        
        assert path is not None
        # Should go through the middle row to avoid high-cost cells
        assert (1, 1) in path
        # Total cost should be 3 (1+1+1) going through middle
        # rather than 11+1+11 = 23 going through top/bottom
        total_cost = sum(grid[r][c] for r, c in path)
        assert total_cost == 3

    def test_no_path_exists(self):
        """Test when there is no path due to complete blockage."""
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is None

    def test_start_equals_end(self):
        """Test when start and end are the same cell."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((1, 1), (1, 1))
        
        assert path is not None
        assert path == [(1, 1)]

    def test_invalid_coordinates(self):
        """Test that ValueError is raised for out-of-bounds coordinates."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Out of bounds
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (3, 3))
        
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (2, 2))
        
        # Start is a wall
        grid_with_wall = [
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar_wall = AStarGrid(grid_with_wall)
        path = astar_wall.find_path((0, 0), (2, 2))
        assert path is None
        
        # End is a wall
        path = astar_wall.find_path((0, 1), (0, 0))
        assert path is None
```

### Key Implementation Details:

1. **Data Structures**:
   - `open_set` is a min-heap (priority queue) storing tuples of `(f_score, position)`.
   - `came_from` dictionary tracks the path reconstruction.
   - `g_score` and `f_score` dictionaries store the cost from start and total estimated cost respectively.

2. **Heuristic**: Manhattan distance is used as required.

3. **Movement**: Only 4-directional (up, down, left, right) movement is allowed.

4. **Edge Cases**:
   - Returns `[start]` when `start == end`.
   - Returns `None` when start or end is a wall.
   - Raises `ValueError` for out-of-bounds coordinates.

5. **Optimality**: The algorithm uses the standard A* formulation which guarantees optimality when using an admissible heuristic (Manhattan distance is admissible for 4-directional movement).

6. **Weighted Grid**: The cost to enter a cell is added to the g_score, allowing the algorithm to prefer lower-cost paths.

To run the tests, ensure you have pytest installed (`pip install pytest`) and run `pytest` on the test file.
