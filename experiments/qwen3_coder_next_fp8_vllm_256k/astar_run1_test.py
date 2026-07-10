from typing import List, Tuple, Optional
import heapq
import math

Coordinate = Tuple[int, int]


class AStarGrid:
    """
    A weighted 2D grid for A* pathfinding.

    Grid cells:
        - Positive integers represent movement cost (1 = flat terrain).
        - 0 represents a wall (impassable).
        - Must be rectangular.

    Attributes:
        grid: 2D list of non-negative integers.
        rows: Number of rows.
        cols: Number of columns.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid.

        Args:
            grid: 2D list of non-negative integers where 0 indicates a wall.

        Raises:
            ValueError: If grid is empty, rectangular rows, or contains negative values.
        """
        if not grid:
            raise ValueError("Grid cannot be empty")

        self.rows = len(grid)
        self.cols = len(grid[0])

        if self.cols == 0:
            raise ValueError("Grid columns cannot be zero")

        # Validate grid structure and values
        for r, row in enumerate(grid):
            if len(row) != self.cols:
                raise ValueError("Grid rows must have equal length")
            for c, cell in enumerate(row):
                if cell < 0:
                    raise ValueError(f"Cell at ({r}, {c}) has negative value: {cell}")

        self.grid = grid

    def _is_valid(self, pos: Coordinate) -> bool:
        """
        Check if position is within grid bounds and not a wall.

        Args:
            pos: (row, col) position.

        Returns:
            True if valid, False otherwise.
        """
        r, c = pos
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r][c] > 0
        return False

    @staticmethod
    def _manhattan(a: Coordinate, b: Coordinate) -> int:
        """
        Manhattan distance between two points.

        Args:
            a: First point (row, col).
            b: Second point (row, col).

        Returns:
            Manhattan distance.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Coordinate, end: Coordinate) -> Optional[List[Coordinate]]:
        """
        Find the optimal path from start to end using A*.

        Args:
            start: (row, col) starting position.
            end: (row, col) target position.

        Returns:
            List of coordinates from start to end (inclusive), or None if no path exists.

        Raises:
            ValueError: If start or end is out-of-bounds or a wall.
        """
        # Validate start and end coordinates
        if not self._is_valid(start):
            raise ValueError(f"Start position {start} is invalid (out-of-bounds or wall)")
        if not self._is_valid(end):
            raise ValueError(f"End position {end} is invalid (out-of-bounds or wall)")

        # Handle trivial case
        if start == end:
            return [start]

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Priority queue: (f_score, counter, (r, c))
        # counter breaks ties for equal f_scores (ensures deterministic behavior)
        counter = 0
        open_set = [(self._manhattan(start, end), counter, start)]

        # came_from[(r, c)] = parent coordinate
        came_from: dict[Coordinate, Coordinate] = {}

        # g_score[(r, c)] = cost from start to (r, c)
        g_score: dict[Coordinate, int] = {start: 0}

        # open_set_hash for O(1) lookup
        open_set_hash = {start}

        while open_set:
            _, _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)

                if not self._is_valid(neighbor):
                    continue

                # Cost to move to neighbor is the neighbor's cell value
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._manhattan(neighbor, end)

                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f_score, counter, neighbor))
                        open_set_hash.add(neighbor)

        # No path found
        return None

import pytest


def test_trivial_same_start_end():
    """Test path when start == end."""
    grid = AStarGrid([[1, 2], [3, 4]])
    path = grid.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]


def test_simple_path():
    """Test basic unweighted path."""
    grid_data = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    grid = AStarGrid(grid_data)
    path = grid.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Path should avoid the wall at (1,1)
    assert (1, 1) not in path


def test_weighted_path():
    """Test path chooses cheaper route."""
    # Cost 100 in the middle; path should go around
    grid_data = [
        [1, 1, 1],
        [1, 100, 1],
        [1, 1, 1]
    ]
    grid = AStarGrid(grid_data)
    path = grid.find_path((0, 0), (2, 2))
    assert path is not None
    # Path could be: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) [cost = 1+1+1+1 = 4]
    # or (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2) [cost = 1+1+1+1 = 4]
    # Or any 4-step path with 4 ones. Not going through center.
    cost = sum(grid_data[r][c] for r, c in path)
    assert cost == 4


def test_no_path():
    """Test when start and end are separated by a wall."""
    grid_data = [
        [1, 1, 1],
        [0, 0, 0],  # Full wall
        [1, 1, 1]
    ]
    grid = AStarGrid(grid_data)
    path = grid.find_path((0, 1), (2, 1))
    assert path is None


def test_out_of_bounds_start():
    """Test ValueError for out-of-bounds start."""
    grid = AStarGrid([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        grid.find_path((-1, 0), (1, 1))


def test_out_of_bounds_end():
    """Test ValueError for out-of-bounds end."""
    grid = AStarGrid([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        grid.find_path((0, 0), (2, 0))


def test_wall_as_end():
    """Test ValueError when end is a wall."""
    grid = AStarGrid([[1, 0], [3, 4]])
    with pytest.raises(ValueError):
        grid.find_path((0, 0), (0, 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])