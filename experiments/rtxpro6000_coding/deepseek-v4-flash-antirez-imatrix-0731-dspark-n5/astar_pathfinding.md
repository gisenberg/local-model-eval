```python
from typing import List, Tuple, Optional
import heapq


class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.

    Grid values represent movement cost (0 = impassable wall, positive int = cost to enter that cell).
    Supports 4-directional movement (up, down, left, right) with Manhattan distance heuristic.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid for pathfinding.

        Args:
            grid: 2D list where grid[row][col] is the movement cost (0 = wall, positive = cost)
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, row: int, col: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _is_wall(self, row: int, col: int) -> bool:
        """Check if a cell is a wall (cost 0)."""
        return self.grid[row][col] == 0

    def _manhattan_distance(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """Calculate Manhattan distance between two cells."""
        return abs(row1 - row2) + abs(col1 - col2)

    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
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
        start_g = 0
        start_h = self._manhattan_distance(start[0], start[1], end[0], end[1])
        start_f = start_g + start_h
        open_heap = [(start_f, start_g, start[0], start[1], None, None)]

        # Track visited nodes with their best g_score and parent
        visited = {}  # (row, col) -> (g_score, parent_row, parent_col)
        visited[(start[0], start[1])] = (0, None, None)

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
            for neighbor_row, neighbor_col in self._get_neighbors(row, col):
                # Calculate new g_score (cost to reach neighbor)
                new_g = g_score + self.grid[neighbor_row][neighbor_col]

                # Check if this path to neighbor is better than previously found
                if (neighbor_row, neighbor_col) not in visited or new_g < visited[(neighbor_row, neighbor_col)][0]:
                    # Update visited with new g_score and parent
                    visited[(neighbor_row, neighbor_col)] = (new_g, row, col)

                    # Calculate f_score for neighbor
                    h_score = self._manhattan_distance(neighbor_row, neighbor_col, end[0], end[1])
                    f_score = new_g + h_score

                    # Add to open heap
                    heapq.heappush(open_heap, (f_score, new_g, neighbor_row, neighbor_col, row, col))

        # No path found
        return None


# Test cases
import pytest

def test_simple_path_uniform_grid():
    """Test pathfinding on a uniform grid with no obstacles."""
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
    # Verify path validity
    for i in range(len(path) - 1):
        row_diff = abs(path[i+1][0] - path[i][0])
        col_diff = abs(path[i+1][1] - path[i][1])
        assert row_diff + col_diff == 1  # 4-directional movement

    # Verify optimality (minimum cost)
    total_cost = sum(grid[r][c] for r, c in path)
    assert total_cost == 5  # 5 cells × cost 1

def test_path_around_obstacles():
    """Test pathfinding around obstacles."""
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
    # Path must avoid walls (cost 0)
    for r, c in path:
        assert grid[r][c] != 0

    # Verify optimality
    total_cost = sum(grid[r][c] for r, c in path)
    # Expected path: (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3) = 6
    assert total_cost == 6

def test_weighted_grid_prefers_lower_cost():
    """Test that pathfinding prefers lower-cost cells."""
    grid = [
        [1, 10, 10, 10],
        [1, 10, 10, 10],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 3))

    assert path is not None
    # Path should go through low-cost cells (cost 1) rather than high-cost (cost 10)
    total_cost = sum(grid[r][c] for r, c in path)
    # Expected optimal path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(2,3) = 6
    assert total_cost == 6

    # Verify path uses low-cost cells
    for r, c in path:
        assert grid[r][c] == 1

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
```