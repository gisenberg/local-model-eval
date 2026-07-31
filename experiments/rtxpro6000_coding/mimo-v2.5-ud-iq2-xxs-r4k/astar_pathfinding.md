Here's the complete implementation with tests:

```python
"""
A* Pathfinding on a Weighted 2D Grid
===================================
Implements the A* algorithm for finding the shortest path on a 2D grid
where each cell has a movement cost. Supports 4-directional movement
and uses Manhattan distance as the heuristic.
"""

from typing import List, Tuple, Optional
import heapq


class AStarGrid:
    """A* pathfinding on a weighted 2D grid.

    Attributes:
        grid: 2D list where grid values represent movement cost.
              0 = impassable wall, positive int = cost to enter that cell.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the A* pathfinding grid.

        Args:
            grid: 2D list where grid values represent movement cost.
                  0 = impassable wall, positive int = cost to enter that cell.

        Raises:
            ValueError: If grid is empty or has inconsistent row lengths.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        # Validate all rows have the same length
        for i, row in enumerate(grid):
            if len(row) != self.cols:
                raise ValueError(f"Row {i} has length {len(row)}, expected {self.cols}")

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find the shortest path from start to end using A* algorithm.

        Args:
            start: Starting position as (row, col).
            end: Ending position as (row, col).

        Returns:
            List of (row, col) coordinates from start to end inclusive,
            or None if no path exists.

        Raises:
            ValueError: If start or end is out of bounds.
        """
        # Validate coordinates are within bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start position {start} is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End position {end} is out of bounds")

        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None

        # Handle edge case: start == end
        if start == end:
            return [start]

        # A* algorithm
        # Priority queue: (f_score, g_score, row, col)
        # f_score = g_score + heuristic
        open_set: list = []
        heapq.heappush(open_set, (0, 0, start[0], start[1]))

        # Track the cost to reach each node
        g_score: dict[Tuple[int, int], int] = {start: 0}

        # Track the parent of each node for path reconstruction
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}

        # Track visited nodes
        closed_set: set[Tuple[int, int]] = set()

        # Directions: up, down, left, right
        directions: list[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            # Get the node with the lowest f_score
            f_score, g_score_current, row, col = heapq.heappop(open_set)

            # If we've already processed this node, skip it
            if (row, col) in closed_set:
                continue

            # Mark as visited
            closed_set.add((row, col))

            # If we've reached the end, reconstruct the path
            if (row, col) == end:
                path: list[Tuple[int, int]] = []
                current: Optional[Tuple[int, int]] = end
                while current is not None:
                    path.append(current)
                    current = came_from.get(current)
                path.reverse()
                return path

            # Explore neighbors
            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc

                # Check if neighbor is within bounds
                if not (0 <= new_row < self.rows and 0 <= new_col < self.cols):
                    continue

                # Check if neighbor is a wall
                if self.grid[new_row][new_col] == 0:
                    continue

                # Check if neighbor is already visited
                if (new_row, new_col) in closed_set:
                    continue

                # Calculate new g_score
                new_g_score = g_score_current + self.grid[new_row][new_col]

                # If we've found a better path to this neighbor
                if (new_row, new_col) not in g_score or new_g_score < g_score[(new_row, new_col)]:
                    g_score[(new_row, new_col)] = new_g_score
                    f_score = new_g_score + self._heuristic((new_row, new_col), end)
                    came_from[(new_row, new_col)] = (row, col)
                    heapq.heappush(open_set, (f_score, new_g_score, new_row, new_col))

        # No path found
        return None

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points.

        Args:
            a: First position as (row, col).
            b: Second position as (row, col).

        Returns:
            Manhattan distance between a and b.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


# =============================================================================
# pytest tests
# =============================================================================

import pytest


class TestAStarGrid:
    """Test suite for AStarGrid pathfinding."""

    def test_simple_path_uniform_grid(self) -> None:
        """Test simple path on uniform grid."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        # Total cost should be 4 (4 moves, each cost 1)
        total_cost = sum(grid[r][c] for r, c in path[1:])  # Exclude start
        assert total_cost == 4

    def test_path_around_obstacles(self) -> None:
        """Test path around obstacles."""
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        # Path should avoid the wall at (1, 1)
        assert (1, 1) not in path
        # Total cost should be 4 (4 moves, each cost 1)
        total_cost = sum(grid[r][c] for r, c in path[1:])
        assert total_cost == 4

    def test_weighted_grid(self) -> None:
        """Test weighted grid where path prefers lower-cost cells."""
        grid = [
            [1, 10, 1],
            [1, 10, 1],
            [1, 1, 1],
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        # Path should prefer the left column (cost 1) over the middle column (cost 10)
        # Optimal path: (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2)
        # Total cost: 1 + 1 + 1 + 1 = 4
        total_cost = sum(grid[r][c] for r, c in path[1:])
        assert total_cost == 4

    def test_no_path_exists(self) -> None:
        """Test no path exists (fully blocked)."""
        grid = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        assert path is None

    def test_start_equals_end(self) -> None:
        """Test start equals end."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((1, 1), (1, 1))
        assert path is not None
        assert path == [(1, 1)]

    def test_invalid_coordinates(self) -> None:
        """Test invalid coordinates."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        astar = AStarGrid(grid)
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (2, 2))
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (3, 3))
```

**Key design decisions:**

1. **Heap tuple structure**: `(f_score, g_score, row, col)` — `g_score` is included as a tiebreaker so that when two nodes have the same `f_score`, the one with the lower `g_score` (closer to start) is explored first.

2. **Cost calculation**: The cost to enter a cell is `grid[row][col]`. The start cell's cost is not included in the total cost since we're already there.

3. **Path reconstruction**: Uses `came_from` dictionary to track parent pointers, then reconstructs the path by following the chain from end to start.

4. **Heuristic**: Manhattan distance `|r1 - r2| + |c1 - c2|` is admissible and consistent for 4-directional movement, guaranteeing optimality.

5. **Edge cases**:
   - `start == end` returns `[start]`
   - Wall at start/end returns `None`
   - Out-of-bounds raises `ValueError`

6. **Tests**: Cover all 6 required scenarios with assertions on both path validity and optimality (total cost).
