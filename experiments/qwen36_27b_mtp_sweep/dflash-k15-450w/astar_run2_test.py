import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Walls are represented by 0. Valid cells have positive weights representing 
    the cost to enter them. Movement is restricted to 4 directions (up, down, left, right).
    """

    def __init__(self, grid: List[List[float]]):
        """
        Initialize the grid for pathfinding.

        Args:
            grid: 2D list of numbers where 0 represents a wall and >0 represents cell weight.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty and rectangular.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not a wall."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] > 0

    def _heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> float:
        """Manhattan distance heuristic (admissible and consistent for 4-directional movement)."""
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.

        Args:
            start: Starting coordinates (row, col).
            end: Target coordinates (row, col).

        Returns:
            List of coordinates from start to end, or None if no path exists.

        Raises:
            ValueError: If start or end is out of bounds or on a wall.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds or on a wall.")

        if start == end:
            return [start]

        # Priority queue stores (f_score, position)
        open_set: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while open_set:
            current_f, current = heapq.heappop(open_set)

            # Lazy deletion: skip if we've already found a better path to this node
            if current_f > g_score.get(current, float('inf')):
                continue

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._is_valid(neighbor):
                    continue

                # Cost to enter neighbor cell
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))

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
    """Test when start and end positions are identical."""
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_no_path_due_to_walls():
    """Test that blocked paths correctly return None."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds_raises_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_start_on_wall_raises_error():
    """Test that starting on a wall raises ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))

def test_weighted_optimal_path():
    """Test that A* chooses the lowest-cost path over the shortest Manhattan distance."""
    grid = [
        [1, 100, 1],
        [1, 100, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Must avoid high-weight cells
    assert (0, 1) not in path
    assert (1, 1) not in path
    # Verify it takes the bottom route (cost = 4 vs 103)
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]