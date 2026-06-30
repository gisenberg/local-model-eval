import heapq
import math
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    The grid is represented as a list of lists where each cell contains an integer.
    A value of 0 represents a wall (impassable), and any positive integer represents
    the movement cost to enter that cell. Coordinates are given as (row, col).
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the AStarGrid with a 2D grid.

        Args:
            grid: 2D list of integers where 0 is a wall and >0 is movement cost.

        Raises:
            ValueError: If grid is empty or rows have inconsistent lengths.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0])
        
        for row in grid:
            if len(row) != self.width:
                raise ValueError("All rows in the grid must have the same length")

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid boundaries."""
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if position contains a wall."""
        r, c = pos
        return self.grid[r][c] == 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic for 4-directional movement."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start: Tuple (row, col) representing the starting position.
            end: Tuple (row, col) representing the target position.

        Returns:
            A list of (row, col) tuples representing the optimal path, 
            or None if no valid path exists.

        Raises:
            ValueError: If start or end is out of bounds or positioned on a wall.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds")
        if self._is_wall(start) or self._is_wall(end):
            raise ValueError("Start or end position is on a wall")

        if start == end:
            return [start]

        # Priority queue: (f_score, tie_breaker, position)
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))

        g_score: Dict[Tuple[int, int], float] = defaultdict(lambda: math.inf)
        g_score[start] = 0
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        counter = 1

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # 4-directional movement: down, up, right, left
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not self._is_valid(neighbor) or self._is_wall(neighbor):
                    continue

                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1

        return None

import pytest

def test_basic_pathfinding():
    """Test standard path around a single wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # 4 moves around the center wall

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_exists():
    """Test when start and end are completely separated by walls."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((2, 2), (1, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (-1, 0))

def test_wall_at_start_or_end_raises_value_error():
    """Test that positioning start/end on walls raises ValueError."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((1, 1), (0, 0))

def test_weighted_optimality():
    """Test that A* correctly chooses the lowest-cost path over shortest steps."""
    # Middle column has high cost (10), outer paths cost 1
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    
    # Verify optimality by calculating total movement cost
    total_cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert total_cost == 4  # Optimal path avoids the 10-cost column entirely
    
    # Ensure no high-cost cell was traversed
    for r, c in path:
        assert astar.grid[r][c] != 10