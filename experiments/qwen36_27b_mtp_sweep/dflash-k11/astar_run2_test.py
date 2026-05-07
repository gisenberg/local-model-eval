import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid Representation:
    - 0: Wall (impassable)
    - >0: Traversal cost (weight) to enter the cell
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the AStarGrid with a 2D grid.
        
        Args:
            grid: A rectangular 2D list of integers where 0 represents walls
                  and positive integers represent traversal costs.
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
        """Check if a cell is a wall."""
        return self.grid[pos[0]][pos[1]] == 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic (admissible & consistent for 4-dir movement)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.
        
        Args:
            start: Starting coordinates (row, col)
            end: Target coordinates (row, col)
            
        Returns:
            A list of coordinates representing the optimal path, or None if unreachable.
            
        Raises:
            ValueError: If start or end is out of bounds or placed on a wall.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds.")
        if self._is_wall(start) or self._is_wall(end):
            raise ValueError("Start or end position is a wall.")
            
        if start == end:
            return [start]

        # Priority queue: (f_score, tie_breaker, position)
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))
        
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        counter = 1  # Tie-breaker to avoid comparing tuples in heapq

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

            # 4-directional movement
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not self._is_valid(neighbor) or self._is_wall(neighbor):
                    continue

                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost

                # Relaxation step
                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    came_from[neighbor] = current
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
    assert len(path) == 5  # Optimal steps in 3x3 grid

def test_start_equals_end():
    """Test when start and end coordinates are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_exists():
    """Test when target is completely blocked by walls."""
    grid = [[1, 0, 1], [1, 0, 1], [1, 0, 1]]
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

def test_wall_at_start_or_end_raises_value_error():
    """Test that placing start/end on a wall raises ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (0, 1))
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 1), (0, 0))

def test_weighted_optimal_path():
    """Test that A* chooses lower-cost path over shorter geometric path."""
    # Direct route: (0,0)->(0,1)->(0,2) costs 10 + 1 = 11
    # Detour route: goes around bottom, costs 1*6 = 6
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    assert (0, 1) not in path  # Must avoid high-weight cell
    assert path[0] == (0, 0)
    assert path[-1] == (0, 2)