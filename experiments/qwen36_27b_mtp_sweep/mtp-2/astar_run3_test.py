import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid representation:
    - 0: Wall (impassable)
    - Positive integer: Cost to enter the cell
    """
    def __init__(self, grid: List[List[int]]) -> None:
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        # 4-directional movement: Right, Left, Down, Up
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def _heuristic(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """Manhattan distance heuristic for 4-directional movement."""
        return abs(r1 - r2) + abs(c1 - c2)

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start: Tuple of (row, col) for starting position.
            end: Tuple of (row, col) for ending position.

        Returns:
            List of (row, col) tuples representing the optimal path,
            or None if no valid path exists.

        Raises:
            ValueError: If start or end coordinates are out of bounds or on a wall (0).
        """
        sr, sc = start
        er, ec = end

        # Validate bounds
        if not (0 <= sr < self.rows and 0 <= sc < self.cols):
            raise ValueError("Start position is out of bounds.")
        if not (0 <= er < self.rows and 0 <= ec < self.cols):
            raise ValueError("End position is out of bounds.")

        # Validate walls
        if self.grid[sr][sc] == 0:
            raise ValueError("Start position is a wall.")
        if self.grid[er][ec] == 0:
            raise ValueError("End position is a wall.")

        # Handle start == end
        if start == end:
            return [start]

        # Priority queue: (f_score, counter, (row, col))
        # Counter breaks ties deterministically
        counter = 0
        open_set = [(self._heuristic(sr, sc, er, ec), counter, start)]

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}

        while open_set:
            f, _, current = heapq.heappop(open_set)
            cr, cc = current

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dr, dc in self.directions:
                nr, nc = cr + dr, cc + dc

                # Check bounds and walls
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr][nc] == 0:
                    continue

                tentative_g = g_score[current] + self.grid[nr][nc]
                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + self._heuristic(nr, nc, er, ec)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, (nr, nc)))

        return None

import pytest

def test_basic_pathfinding():
    """Test simple path on an open grid."""
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
    assert len(path) == 5  # Optimal length for 3x3 grid

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_no_path_exists():
    """Test when end is completely surrounded by walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_positions_raise_value_error():
    """Test that starting or ending on a wall raises ValueError."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (0, 1))
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 1), (0, 0))

def test_weighted_optimality():
    """Test that A* chooses the cheaper path over the shorter Manhattan path."""
    # Direct path cost: 100 + 1 = 101
    # Detour path cost: 1 + 1 + 1 + 1 = 4
    grid = [
        [1, 100, 1],
        [1,   1, 1],
        [1,   1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is not None
    # Optimal path must detour through row 1 to avoid the high-cost cell (0, 1)
    assert (1, 0) in path
    assert (1, 1) in path
    assert (0, 1) not in path