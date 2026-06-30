import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid cells with value 0 are walls (impassable).
    Cells with value > 0 are walkable, with the value representing the traversal cost.
    Movement is restricted to 4 directions (up, down, left, right).
    """

    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic (admissible for 4-directional movement)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.
        
        Args:
            start: (row, col) tuple for starting position.
            end: (row, col) tuple for target position.
            
        Returns:
            List of (row, col) tuples representing the optimal path, 
            or None if no valid path exists.
            
        Raises:
            ValueError: If start/end are out of bounds or positioned on a wall.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds.")
            
        # Validate walls
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall.")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall.")
            
        # Handle start == end
        if start == end:
            return [start]

        # Priority queue: (f_score, tie_breaker, (row, col))
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))
        
        g_score = {start: 0}
        came_from = {}
        counter = 1  # Ensures stable ordering when f_scores tie

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

            for dr, dc in self.directions:
                nr, nc = current[0] + dr, current[1] + dc
                
                # Check bounds and walls
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != 0:
                    weight = self.grid[nr][nc]
                    tentative_g = g_score[current] + weight

                    if tentative_g < g_score.get((nr, nc), float('inf')):
                        came_from[(nr, nc)] = current
                        g_score[(nr, nc)] = tentative_g
                        f_score = tentative_g + self._heuristic((nr, nc), end)
                        heapq.heappush(open_set, (f_score, counter, (nr, nc)))
                        counter += 1

        return None

import pytest

def test_simple_path():
    """Basic pathfinding on an open grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

def test_start_equals_end():
    """Path when start and end are the same cell."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_blocked():
    """Returns None when walls completely block the target."""
    grid = [[1, 0, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds_start():
    """Raises ValueError for out-of-bounds start position."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))

def test_wall_at_start():
    """Raises ValueError when start or end is on a wall."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))

def test_weighted_optimal_path():
    """Verifies A* chooses the lower-cost detour over a direct high-cost route."""
    grid = [
        [1, 10, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    # Direct path through (0,1) costs 10. Detour costs 4. A* must avoid (0,1).
    assert (0, 1) not in path