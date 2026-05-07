import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    The grid uses 0 to represent walls (impassable) and positive integers 
    to represent the cost to enter a cell. Movement is restricted to 4 directions.
    """
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid.
        
        Args:
            grid: 2D list of integers where 0 represents a wall and 
                  positive integers represent movement costs.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: Tuple (row, col) of the starting position.
            end: Tuple (row, col) of the target position.
            
        Returns:
            A list of (row, col) tuples representing the path from start to end,
            or None if no path exists.
            
        Raises:
            ValueError: If start or end is out of bounds or on a wall.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end coordinates are out of bounds or on a wall.")
            
        if start == end:
            return [start]

        # Priority queue stores (f_score, counter, (row, col))
        # counter breaks ties to prevent tuple comparison errors
        open_set = [(0, 0, start)]
        counter = 0
        g_score = {start: 0}
        came_from = {start: None}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)

                if not self._is_valid(neighbor):
                    continue

                # Cost to enter the neighbor cell
                move_cost = self.grid[nr][nc]
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return None

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within bounds and not a wall."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != 0

    def _heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Manhattan distance heuristic (admissible for 4-directional movement)."""
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from start to end using came_from map."""
        path = [current]
        while came_from[current] is not None:
            current = came_from[current]
            path.append(current)
        return path[::-1]


# ========================
# Pytest Test Suite
# ========================
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
    assert len(path) == 5  # Minimum steps for 3x3 grid

def test_start_equals_end():
    """Test when start and end coordinates are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_returns_none():
    """Test that blocked paths correctly return None."""
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

def test_start_or_end_on_wall_raises_value_error():
    """Test that start/end on a wall raises ValueError."""
    grid = [
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 1), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 1))

def test_weighted_grid_optimal_path():
    """Test that A* chooses the lowest-cost path, not just shortest steps."""
    grid = [
        [1, 100, 1],
        [1, 100, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    
    # Verify path avoids high-cost cells
    for r, c in path:
        assert grid[r][c] != 100
        
    # Calculate total movement cost (excluding start cell)
    cost = sum(grid[r][c] for r, c in path[1:])
    assert cost == 4  # Optimal path goes around the 100s