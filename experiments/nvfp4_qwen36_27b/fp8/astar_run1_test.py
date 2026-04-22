import heapq
import math
from typing import List, Tuple, Optional
from collections import defaultdict


class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid representation:
    - 0: Wall (impassable)
    - Positive integers: Traversal cost/weight for entering the cell
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid and precompute dimensions.
        
        Args:
            grid: 2D list where 0 represents walls and positive integers represent weights.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within bounds and not a wall."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic (admissible & consistent for 4-directional movement)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.
        
        Args:
            start: Starting coordinates (row, col)
            end: Target coordinates (row, col)
            
        Returns:
            List of coordinates from start to end, or None if no path exists.
            
        Raises:
            ValueError: If start or end is out of bounds or positioned on a wall.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds or on a wall.")
        if start == end:
            return [start]

        # Priority queue stores (f_score, counter, position)
        # counter breaks ties and prevents tuple comparison errors
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))

        g_score: dict = defaultdict(lambda: math.inf)
        g_score[start] = 0
        came_from: dict = {}
        counter = 1

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dr, dc in self.directions:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._is_valid(neighbor):
                    continue

                # Cost to enter neighbor cell
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]
                
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1

        return None

import pytest

def test_basic_pathfinding():
    """Test standard pathfinding on an open grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Optimal length for 3x3 grid

def test_start_equals_end():
    """Test when start and end coordinates are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_exists():
    """Test grid where target is completely blocked by walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_wall_positions_raise_value_error():
    """Test that starting or ending on a wall raises ValueError."""
    grid = [
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 1), (1, 1))  # Start on wall
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 1))  # End on wall

def test_weighted_optimality():
    """Test that A* chooses the cheaper path over the shorter one."""
    # Direct path through center costs 100, perimeter costs 4
    grid = [
        [1, 1, 1],
        [1, 100, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert (1, 1) not in path  # Must avoid expensive center cell
    # Verify it takes one of the two optimal perimeter routes
    expected = [
        [(0,0), (0,1), (0,2), (1,2), (2,2)],
        [(0,0), (1,0), (2,0), (2,1), (2,2)]
    ]
    assert path in expected