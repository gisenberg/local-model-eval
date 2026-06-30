import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid representation:
    - 0.0 represents walls/impassable cells.
    - Positive numbers represent walkable cells with associated movement costs.
    """

    def __init__(self, grid: List[List[float]]) -> None:
        """
        Initialize the grid for pathfinding.
        
        Args:
            grid: 2D list of cell costs. 0 indicates a wall.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        # 4-directional movement: Right, Left, Down, Up
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid boundaries."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a wall (cost == 0)."""
        r, c = pos
        return self.grid[r][c] == 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic for 4-directional grid movement."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: Tuple (row, col) of the starting position.
            end: Tuple (row, col) of the target position.
            
        Returns:
            List of (row, col) tuples representing the optimal path, 
            or None if no valid path exists.
            
        Raises:
            ValueError: If start or end is out of grid bounds.
        """
        # Validate bounds
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds.")
            
        # Handle wall cases
        if self._is_wall(start) or self._is_wall(end):
            return None
            
        # Handle identical start/end
        if start == end:
            return [start]

        # Priority queue: (f_score, counter, position)
        # Counter ensures stable sorting and avoids tuple comparison errors
        open_set: list = []
        initial_f = self._heuristic(start, end)
        heapq.heappush(open_set, (initial_f, 0, start))

        g_score: dict[Tuple[int, int], float] = {start: 0.0}
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        counter = 1

        while open_set:
            f, _, current = heapq.heappop(open_set)

            # Lazy deletion: skip stale entries
            if current in g_score and f > g_score[current] + self._heuristic(current, end):
                continue

            # Goal reached
            if current == end:
                path: List[Tuple[int, int]] = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Explore neighbors
            for dr, dc in self.directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not self._is_valid(neighbor) or self._is_wall(neighbor):
                    continue

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

def test_basic_pathfinding():
    """Test standard path on an unweighted grid."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (0, 0))
        
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_start_or_end_is_wall():
    """Test that starting or ending on a wall returns None."""
    grid = [
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 1), (2, 2)) is None
    assert astar.find_path((0, 0), (0, 1)) is None

def test_weighted_cells_prefer_lower_cost():
    """Test that A* correctly routes around high-cost cells."""
    grid = [
        [1, 10, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is not None
    # Direct path costs 12 (1+10+1), detour costs 4 (1+1+1+1)
    assert (0, 1) not in path
    assert len(path) == 5  # (0,0)->(1,0)->(1,1)->(1,2)->(0,2)

def test_no_path_returns_none():
    """Test that completely blocked paths return None."""
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((1, 1), (0, 0)) is None