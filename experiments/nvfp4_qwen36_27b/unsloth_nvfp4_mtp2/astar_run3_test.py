import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """A* pathfinding algorithm on a weighted 2D grid.

    Movement is 4-directional. Cell values represent the cost to enter that cell.
    A value of 0 represents an impassable wall.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid.

        Args:
            grid: 2D list where 0 represents a wall and positive integers
                  represent the cost to enter the cell.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within bounds and not a wall."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.

        Args:
            start: Starting coordinates (row, col).
            end: Ending coordinates (row, col).

        Returns:
            List of coordinates from start to end, or None if no path exists.

        Raises:
            ValueError: If start or end are out of bounds or on walls.
        """
        if not self._is_valid(start):
            raise ValueError("Start position is out of bounds or on a wall.")
        if not self._is_valid(end):
            raise ValueError("End position is out of bounds or on a wall.")

        if start == end:
            return [start]

        # Priority queue stores (f_score, counter, position)
        # Counter ensures stable sorting when f_scores are equal
        open_set: List[Tuple[int, int, Tuple[int, int]]] = [(0, 0, start)]
        counter = 1
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        g_score: Dict[Tuple[int, int], int] = {start: 0}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._is_valid(neighbor):
                    continue

                # Cost to move into the neighbor cell
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1

        return None

import pytest

def test_basic_path():
    """Test standard pathfinding around a wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_no_path():
    """Test when start and end are completely separated by walls."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) is None

def test_out_of_bounds():
    """Test ValueError for coordinates outside grid dimensions."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_at_start_or_end():
    """Test ValueError when start or end lies on a wall."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((1, 0), (0, 1))

def test_weighted_optimal_path():
    """Test that A* chooses the lowest-cost path, not just fewest steps."""
    # Direct path cost: 10 + 1 = 11
    # Detour path cost: 1 + 1 + 1 + 1 = 4
    grid = [
        [1, 10, 1],
        [1,  1, 1],
        [1,  1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert (0, 1) not in path  # Should avoid high-cost cell
    assert path == [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]