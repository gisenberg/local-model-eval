import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """A* pathfinding on a weighted 2D grid.

    The grid uses 0 to represent walls (impassable cells) and positive
    integers to represent the movement cost of traversing a cell.
    Movement is 4-directional (up, down, left, right).
    The Manhattan distance is used as the admissible heuristic.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the grid.

        Args:
            grid: 2D list where 0 = wall, positive int = movement cost.

        Raises:
            ValueError: If the grid is empty.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty")
        self.grid: List[List[int]] = grid
        self.rows: int = len(grid)
        self.cols: int = len(grid[0])

    def find_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """Find the optimal path from start to end using A*.

        The cost of a path is the sum of the weights of all cells visited
        except the start cell.

        Args:
            start: (row, col) starting position.
            end: (row, col) ending position.

        Returns:
            List of (row, col) tuples representing the path from start
            to end inclusive, or None if no path exists.

        Raises:
            ValueError: If start or end is out of bounds or on a wall.
        """
        self._validate(start, "start")
        self._validate(end, "end")

        if start == end:
            return [start]

        counter = 0
        open_set: List[Tuple[float, int, Tuple[int, int]]] = [
            (0, counter, start)
        ]
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: dict[Tuple[int, int], int] = {start: 0}

        while open_set:
            current_f, _, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct(came_from, current)

            for nr, nc in self._neighbors(current):
                tentative_g = g_score[current] + self.grid[nr][nc]

                if tentative_g < g_score.get((nr, nc), float("inf")):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    f = tentative_g + self._heuristic((nr, nc), end)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, (nr, nc)))

        return None

    def _validate(self, pos: Tuple[int, int], name: str) -> None:
        """Validate that a position is in bounds and not a wall.

        Args:
            pos: (row, col) position to validate.
            name: Descriptive name for error messages.

        Raises:
            ValueError: If position is out of bounds or on a wall.
        """
        r, c = pos
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(f"{name} ({r}, {c}) is out of bounds")
        if self.grid[r][c] == 0:
            raise ValueError(f"{name} ({r}, {c}) is on a wall")

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance between two positions (admissible heuristic).

        Args:
            a: First (row, col) position.
            b: Second (row, col) position.

        Returns:
            Manhattan distance |a_row - b_row| + |a_col - b_col|.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get 4-directional valid neighbors of a position.

        A neighbor is valid if it is within bounds and not a wall (0).

        Args:
            pos: Current (row, col) position.

        Returns:
            List of (row, col) tuples for valid neighboring cells.
        """
        r, c = pos
        neighbors: List[Tuple[int, int]] = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != 0:
                neighbors.append((nr, nc))
        return neighbors

    def _reconstruct(
        self,
        came_from: dict[Tuple[int, int], Tuple[int, int]],
        current: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """Reconstruct the path by tracing back through the came_from map.

        Args:
            came_from: Dictionary mapping each node to its predecessor.
            current: The end position.

        Returns:
            List of (row, col) tuples from start to end.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

import pytest

def test_simple_path():
    """Test basic pathfinding on a uniform 3×3 grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Manhattan distance is 4 → optimal path has 5 cells
    assert len(path) == 5

def test_weighted_optimal_path():
    """Test that A* finds the optimal (lowest-cost) path on a weighted grid."""
    grid = [
        [1, 2, 1],
        [1, 1, 1],
        [1, 2, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Optimal path routes through cheap (1,1) cell, avoiding cost-2 cells
    assert (1, 1) in path
    # Total cost = 1+1+1+1 = 4 (excluding start)
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4

def test_no_path():
    """Test that A* returns None when no path exists."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    astar = AStarGrid(grid)
    # Column 1 is all walls, blocking any route from (0,0) to (0,2)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_start_equals_end():
    """Test that start==end returns a single-element path."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_out_of_bounds_raises():
    """Test that out-of-bounds positions raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (0, 0))

def test_wall_raises():
    """Test that positions on walls raise ValueError."""
    grid = [[1, 0], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (0, 1))