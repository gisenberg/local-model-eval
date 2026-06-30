import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    - 0 represents walls/impassable cells.
    - Positive integers represent traversal costs (node weights).
    - Uses 4-directional movement, Manhattan heuristic, and heapq for priority queue.
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid.
        
        :param grid: 2D list of integers where 0 is a wall and >0 is a traversal cost.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic (admissible for 4-directional movement)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.
        
        :param start: (row, col) tuple
        :param end: (row, col) tuple
        :return: List of (row, col) tuples representing the optimal path, or None if unreachable.
        :raises ValueError: If start or end coordinates are out of grid bounds.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds")

        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Handle start == end
        if start == end:
            return [start]

        # Priority queue: (f_score, counter, (r, c))
        # Counter ensures FIFO ordering for equal f_scores and avoids tuple comparison errors
        counter = 0
        open_set = [(self._heuristic(start, end), counter, start)]
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            # Goal reached
            if current == end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            # Explore 4-directional neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = current[0] + dr, current[1] + dc
                
                # Check bounds and walls
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != 0:
                    tentative_g = g_score[current] + self.grid[nr][nc]
                    
                    # If this path to neighbor is better, update and push to queue
                    if tentative_g < g_score.get((nr, nc), float('inf')):
                        g_score[(nr, nc)] = tentative_g
                        f_score = tentative_g + self._heuristic((nr, nc), end)
                        came_from[(nr, nc)] = current
                        counter += 1
                        heapq.heappush(open_set, (f_score, counter, (nr, nc)))

        # Queue exhausted without reaching end
        return None

import pytest

def test_basic_open_grid():
    """Test pathfinding on a uniform cost grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

def test_weighted_optimal_route():
    """Verify A* chooses lower cost over shorter step count."""
    grid = [
        [1, 10, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    # Direct path costs 11, detour costs 4. A* must avoid (0,1)
    assert (0, 1) not in path
    assert len(path) == 5

def test_start_equals_end():
    """Handle start == end edge case."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_wall_at_start_or_end():
    """Return None when start or end is blocked."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (1, 1)) is None
    assert astar.find_path((1, 0), (0, 0)) is None

def test_out_of_bounds_raises_value_error():
    """Raise ValueError for invalid coordinates."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 1))

def test_unreachable_destination():
    """Return None when destination is completely walled off."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None