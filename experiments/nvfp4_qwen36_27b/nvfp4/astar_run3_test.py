import heapq
from typing import Optional, List, Tuple, Sequence

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid representation:
    - 0: Wall (impassable)
    - Positive integers: Traversable cells with associated movement cost
    """

    def __init__(self, grid: Sequence[Sequence[int]]):
        """
        Initialize the grid.
        
        :param grid: 2D sequence of integers representing the map.
        """
        self.grid = [list(row) for row in grid]
        self.rows = len(self.grid)
        self.cols = len(self.grid[0]) if self.rows > 0 else 0

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, r: int, c: int) -> bool:
        """Check if a cell is a wall."""
        return self.grid[r][c] == 0

    def _heuristic(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """Manhattan distance heuristic for 4-directional movement."""
        return abs(r1 - r2) + abs(c1 - c2)

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.
        
        :param start: (row, col) tuple for the starting position.
        :param end: (row, col) tuple for the target position.
        :return: List of (row, col) tuples representing the optimal path, or None if unreachable.
        :raises ValueError: If start or end coordinates are out of bounds.
        """
        sr, sc = start
        er, ec = end

        if not self._is_valid(sr, sc) or not self._is_valid(er, ec):
            raise ValueError("Start or end coordinates are out of bounds.")
        if self._is_wall(sr, sc) or self._is_wall(er, ec):
            raise ValueError("Start or end position is a wall.")

        if start == end:
            return [start]

        # Priority queue: (f_score, counter, (r, c))
        # Counter ensures stable ordering when f_scores are equal
        counter = 0
        open_set = [(self._heuristic(sr, sc, er, ec), counter, start)]
        g_score = {start: 0}
        came_from = {}
        closed_set = set()

        # 4-directional movement: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            f, _, current = heapq.heappop(open_set)
            cr, cc = current

            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            if current in closed_set:
                continue
            closed_set.add(current)

            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc
                neighbor = (nr, nc)

                if not self._is_valid(nr, nc) or self._is_wall(nr, nc) or neighbor in closed_set:
                    continue

                # Cost to enter the neighbor cell
                move_cost = self.grid[nr][nc]
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(nr, nc, er, ec)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))
                    came_from[neighbor] = current

        return None


# ==================== PYTEST TESTS ====================

import pytest

def test_simple_path():
    """Test basic pathfinding on an unweighted grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (2, 2)
    assert len(path) == 5  # Optimal length for 3x3 grid

def test_path_around_wall():
    """Test that path correctly navigates around a central wall."""
    grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5  # Must detour around the wall

def test_start_equals_end():
    """Test that start == end returns a single-element path."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path():
    """Test that unreachable targets return None."""
    grid = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_out_of_bounds():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_weighted_optimal():
    """Test that A* selects the lowest-cost path in a weighted grid."""
    # Grid layout:
    # 1  1  1
    # 1  0  1
    # 1 10  1
    # Top route cost: 1+1+1+1 = 4
    # Bottom route cost: 1+1+10+1 = 13
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 10, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Calculate path cost (excluding start cell, matching g_score initialization)
    cost = sum(grid[r][c] for r, c in path[1:])
    assert cost == 4, f"Expected optimal cost 4, got {cost}"