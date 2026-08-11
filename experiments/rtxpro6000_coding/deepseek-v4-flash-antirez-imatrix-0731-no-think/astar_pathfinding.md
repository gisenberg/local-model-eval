```python
from typing import List, Optional, Tuple
import heapq


class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.

    Grid values represent movement cost (0 = impassable wall, positive int = cost to enter that cell).
    Supports 4-directional movement (up, down, left, right) — no diagonals.
    Uses Manhattan distance as heuristic.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid for pathfinding.

        Args:
            grid: 2D list where grid[r][c] is movement cost (0 = wall, positive = cost)
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, row: int, col: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _is_wall(self, row: int, col: int) -> bool:
        """Check if cell is a wall (cost 0)."""
        return self.grid[row][col] == 0

    def _manhattan_distance(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """Calculate Manhattan distance between two cells."""
        return abs(row1 - row2) + abs(col1 - col2)

    def _neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid 4-directional neighbors of a cell."""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self._is_valid(new_row, new_col) and not self._is_wall(new_row, new_col):
                neighbors.append((new_row, new_col))
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path from start to end using A* algorithm.

        Args:
            start: (row, col) starting coordinates
            end: (row, col) target coordinates

        Returns:
            List of (row, col) coordinates from start to end inclusive, or None if no path exists.

        Raises:
            ValueError: If start or end coordinates are out of bounds
        """
        # Validate coordinates
        if not self._is_valid(start[0], start[1]) or not self._is_valid(end[0], end[1]):
            raise ValueError("Start or end coordinates are out of bounds")

        # Check if start or end is a wall
        if self._is_wall(start[0], start[1]) or self._is_wall(end[0], end[1]):
            return None

        # Handle start == end case
        if start == end:
            return [start]

        # Initialize open set with start node
        # Each entry: (f_score, g_score, row, col, parent_row, parent_col)
        # f_score = g_score + heuristic
        start_g = self.grid[start[0]][start[1]]
        start_h = self._manhattan_distance(start[0], start[1], end[0], end[1])
        start_f = start_g + start_h

        open_heap = [(start_f, start_g, start[0], start[1], None, None)]
        heapq.heapify(open_heap)

        # Track visited nodes and their best g_score
        visited = {}  # (row, col) -> (g_score, parent_row, parent_col)
        visited[(start[0], start[1])] = (start_g, None, None)

        while open_heap:
            f_score, g_score, row, col, parent_row, parent_col = heapq.heappop(open_heap)

            # Skip if we've found a better path to this node
            if (row, col) in visited and g_score > visited[(row, col)][0]:
                continue

            # Check if we reached the end
            if (row, col) == end:
                # Reconstruct path
                path = []
                current_row, current_col = row, col
                while current_row is not None:
                    path.append((current_row, current_col))
                    if (current_row, current_col) in visited:
                        _, parent_row, parent_col = visited[(current_row, current_col)]
                        current_row, current_col = parent_row, parent_col
                    else:
                        break
                path.reverse()
                return path

            # Explore neighbors
            for next_row, next_col in self._neighbors(row, col):
                # Calculate new g_score (cost to reach neighbor)
                new_g = g_score + self.grid[next_row][next_col]

                # Check if this path to neighbor is better than previously found
                if (next_row, next_col) not in visited or new_g < visited[(next_row, next_col)][0]:
                    # Update visited
                    visited[(next_row, next_col)] = (new_g, row, col)

                    # Calculate f_score for neighbor
                    h = self._manhattan_distance(next_row, next_col, end[0], end[1])
                    f = new_g + h

                    # Add to open heap
                    heapq.heappush(open_heap, (f, new_g, next_row, next_col, row, col))

        # No path found
        return None


# Test cases
import pytest

def test_simple_path_uniform_grid():
    """Test simple path on uniform grid."""
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
    # Path should be optimal (minimum cost)
    # Each step costs 1, so total cost = len(path) - 1 (start cell cost not counted)
    # For 3x3 grid, shortest path length is 4 cells (3 steps)
    assert len(path) == 4
    # Verify path validity
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1  # 4-directional movement

def test_path_around_obstacles():
    """Test path around obstacles."""
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
    # Path must avoid walls (0 values)
    for row, col in path:
        assert grid[row][col] != 0
    # Path should be optimal
    # Must go around the wall, so minimum steps = 5 (start to (0,1) to (0,2) to (0,3) to (1,3) to (2,3))
    assert len(path) == 5

def test_weighted_grid_prefers_lower_cost():
    """Test weighted grid prefers lower-cost cells."""
    grid = [
        [1, 10, 10, 10],
        [1, 10, 10, 10],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 3))

    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 3)

    # Calculate total cost of path
    total_cost = sum(grid[row][col] for row, col in path[1:])  # exclude start cell
    # Optimal path should go down through low-cost cells (1s)
    # Path: (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2) -> (2,3)
    # Cost = 1 + 1 + 1 + 1 + 1 = 5
    assert total_cost == 5
    # Verify path uses low-cost cells
    for row, col in path:
        assert grid[row][col] == 1

def test_no_path_exists_fully_blocked():
    """Test no path exists when fully blocked."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Block all paths by making walls
    grid[0][1] = 0
    grid[1][0] = 0
    grid[1][2] = 0
    grid[2][1] = 0

    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_start_equals_end():
    """Test when start equals end."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))

    assert path == [(1, 1)]

def test_invalid_coordinates():
    """Test invalid coordinates raise ValueError."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)

    # Out of bounds start
    with pytest.raises(ValueError):
        astar.find_path((5, 5), (0, 0))

    # Out of bounds end
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))

    # Both out of bounds
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))

def test_start_or_end_is_wall():
    """Test when start or end is a wall."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)

    # Start is wall
    assert astar.find_path((0, 0), (1, 1)) is None

    # End is wall
    assert astar.find_path((1, 1), (0, 0)) is None

    # Both are walls
    assert astar.find_path((0, 0), (0, 0)) is None
```