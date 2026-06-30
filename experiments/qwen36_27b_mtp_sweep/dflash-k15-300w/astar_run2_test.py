import heapq
import itertools
from typing import List, Tuple, Optional, Dict, Set

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid Representation:
    - 0: Wall (impassable)
    - >0: Walkable cell with integer traversal cost
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid for pathfinding.
        
        Args:
            grid: 2D list of integers representing the map.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        # 4-directional movement: right, left, down, up
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within bounds and not a wall."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic (admissible & consistent for 4-dir grids)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.
        
        Args:
            start: Starting coordinates (row, col).
            end: Target coordinates (row, col).
            
        Returns:
            List of coordinates representing the optimal path, or None if unreachable.
            
        Raises:
            ValueError: If start or end is out of bounds or positioned on a wall.
        """
        # Handle start == end case
        if start == end:
            if not self._is_valid(start):
                raise ValueError("Start position is out of bounds or is a wall.")
            return [start]

        # Validate inputs
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds or is a wall.")

        # Priority queue stores: (f_score, tie_breaker, position)
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        counter = itertools.count()
        start_f = self._heuristic(start, end)
        heapq.heappush(open_set, (start_f, next(counter), start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        closed_set: Set[Tuple[int, int]] = set()

        while open_set:
            f, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == end:
                # Reconstruct path
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

                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, next(counter), neighbor))

        return None

import pytest

def test_basic_pathfinding():
    """Test standard pathfinding on an open grid."""
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
    # Manhattan distance is 4, so optimal path length is 5 nodes
    assert len(path) == 5

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_exists():
    """Test when target is completely surrounded by walls."""
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
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_position_raises_value_error():
    """Test that starting or ending on a wall raises ValueError."""
    grid = [
        [1, 0],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 1), (1, 1))
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (0, 1))

def test_weighted_optimal_path():
    """Test that A* chooses lowest cost path over shortest step count."""
    grid = [
        [1, 10, 10, 1],
        [1,  0,  0, 1],
        [1,  1,  1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 3))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (0, 3)
    
    # Should avoid the expensive top row (cost 10 each)
    assert (0, 1) not in path
    assert (0, 2) not in path
    
    # Verify total traversal cost matches optimal route
    cost = sum(grid[r][c] for r, c in path[1:])
    assert cost == 7  # 1+1+1+1+1+1+1