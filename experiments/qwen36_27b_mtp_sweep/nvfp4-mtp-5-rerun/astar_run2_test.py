import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """A* pathfinding algorithm on a weighted 2D grid.
    
    Supports 4-directional movement, Manhattan heuristic, and weighted cells.
    Walls are represented by 0, while positive numbers represent traversal costs.
    """

    def __init__(self, grid: List[List[float]]):
        """
        Initialize the grid.
        
        Args:
            grid: 2D list where 0 represents walls and positive numbers represent traversal costs.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if coordinates are within grid bounds."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if a cell is a wall (weight == 0)."""
        r, c = pos
        return self.grid[r][c] == 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: Tuple (row, col) of starting position.
            end: Tuple (row, col) of target position.
            
        Returns:
            List of (row, col) tuples representing the optimal path, or None if unreachable.
            
        Raises:
            ValueError: If start/end are out of bounds or positioned on walls.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds.")
        if self._is_wall(start) or self._is_wall(end):
            raise ValueError("Start or end position is a wall.")

        if start == end:
            return [start]

        # Priority queue stores: (f_score, tie_breaker_counter, g_score, position)
        counter = 0
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, end), counter, 0, start))
        
        g_scores: Dict[Tuple[int, int], float] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_set:
            _, _, current_g, current = heapq.heappop(open_set)

            # Skip stale entries where a better path to this node was already found
            if current_g > g_scores[current]:
                continue

            if current == end:
                # Reconstruct path
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                return path[::-1]

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._is_valid(neighbor) or self._is_wall(neighbor):
                    continue

                weight = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = current_g + weight

                if tentative_g < g_scores.get(neighbor, float('inf')):
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, tentative_g, neighbor))
                    came_from[neighbor] = current

        return None

import pytest

def test_basic_pathfinding():
    """Test standard pathfinding on an unweighted grid."""
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
    # Manhattan distance is 4 steps, so path length should be 5 nodes
    assert len(path) == 5

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
        astar.find_path((0, 0), (2, 2))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (0, 0))

def test_wall_at_start_or_end_raises_value_error():
    """Test that walls at start/end positions raise ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (0, 1))
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 1), (0, 0))

def test_unreachable_path_returns_none():
    """Test that completely blocked paths return None."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_weighted_optimal_path():
    """Test that A* correctly chooses the lowest-cost path over shortest distance."""
    # Direct path costs 100, detour costs 4
    grid = [
        [1, 100, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (0, 2)
    # Verify it avoided the high-cost cell
    assert (0, 1) not in path
    # Verify exact optimal route
    assert path == [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]