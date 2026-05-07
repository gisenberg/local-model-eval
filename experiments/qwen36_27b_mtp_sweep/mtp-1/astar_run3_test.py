import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """A* pathfinding algorithm on a weighted 2D grid."""

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid.

        Args:
            grid: 2D list where 0 represents a wall and positive integers
                  represent the cost to enter the cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.

        Args:
            start: (row, col) tuple for the starting position.
            end: (row, col) tuple for the target position.

        Returns:
            A list of (row, col) tuples representing the optimal path,
            or None if no path exists.

        Raises:
            ValueError: If start/end are out of bounds or located on a wall.
        """
        self._validate_position(start, "Start")
        self._validate_position(end, "End")

        if start == end:
            return [start]

        # Priority queue stores (f_score, tie_breaker, position)
        pq: List[Tuple[float, int, Tuple[int, int]]] = [(0, 0, start)]
        counter = 0
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while pq:
            _, _, current = heapq.heappop(pq)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                if not (0 <= nx < self.rows and 0 <= ny < self.cols):
                    continue
                if self.grid[nx][ny] == 0:
                    continue

                tentative_g = g_score[current] + self.grid[nx][ny]
                if tentative_g < g_score.get((nx, ny), float('inf')):
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g
                    f = tentative_g + self._manhattan((nx, ny), end)
                    counter += 1
                    heapq.heappush(pq, (f, counter, (nx, ny)))

        return None

    def _validate_position(self, pos: Tuple[int, int], name: str) -> None:
        """Check if position is within bounds and not a wall."""
        if not (0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols):
            raise ValueError(f"{name} position is out of bounds.")
        if self.grid[pos[0]][pos[1]] == 0:
            raise ValueError(f"{name} position is a wall.")

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _reconstruct_path(came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from start to end using came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

import pytest

def test_basic_path():
    """Test standard pathfinding around a wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Must go around the center wall

def test_start_equals_end():
    """Test when start and end positions are identical."""
    path = AStarGrid([[1]]).find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_unreachable():
    """Test when end is completely blocked by walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    assert AStarGrid(grid).find_path((0, 0), (0, 2)) is None

def test_out_of_bounds_start():
    """Test ValueError for out-of-bounds start position."""
    with pytest.raises(ValueError, match="out of bounds"):
        AStarGrid([[1, 1], [1, 1]]).find_path((-1, 0), (0, 0))

def test_wall_at_end():
    """Test ValueError when end position is a wall."""
    with pytest.raises(ValueError, match="wall"):
        AStarGrid([[1, 0], [1, 1]]).find_path((0, 0), (0, 1))

def test_weighted_optimal():
    """Test that A* chooses the lowest-cost path over the shortest distance."""
    grid = [
        [1, 10, 1],  # Direct path costs 10 + 1 = 11
        [1,  1, 1],  # Detour costs 1 + 1 + 1 + 1 = 4
        [1,  1, 1]
    ]
    path = AStarGrid(grid).find_path((0, 0), (0, 2))
    assert path is not None
    assert (0, 1) not in path, "Algorithm should avoid the high-weight cell"