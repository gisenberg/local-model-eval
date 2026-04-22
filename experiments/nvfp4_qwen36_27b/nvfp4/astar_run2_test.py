import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid conventions:
    - 0 represents a wall (impassable)
    - Positive integers represent traversal costs to enter that cell
    - Movement is restricted to 4 directions (up, down, left, right)
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the grid and validate dimensions."""
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        # 4-directional movement: (row_delta, col_delta)
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic. Admissible and consistent for 4-directional grids."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within bounds and not a wall."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: Starting coordinates (row, col)
            end: Target coordinates (row, col)
            
        Returns:
            List of coordinates representing the optimal path, or None if unreachable.
            
        Raises:
            ValueError: If start or end is out of bounds or positioned on a wall.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds or on a wall.")
            
        if start == end:
            return [start]

        # Priority queue stores (f_score, position)
        open_set: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_set:
            current_f, current = heapq.heappop(open_set)

            # Lazy deletion: skip if we've already found a better path to this node
            if current_f > g_score.get(current, float('inf')):
                continue

            if current == end:
                # Reconstruct path by backtracking through came_from
                path: List[Tuple[int, int]] = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dr, dc in self.directions:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._is_valid(neighbor):
                    continue

                # Cost to enter the neighbor cell
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))

        return None

import pytest

def test_basic_path():
    """Test standard pathfinding on an unweighted grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Manhattan distance is 4, so optimal path length is 5 nodes
    assert len(path) == 5

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_no_path_exists():
    """Test when end is completely blocked by walls."""
    grid = [[1, 0, 1], [1, 0, 1], [1, 0, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_wall_positions_raise_value_error():
    """Test that starting or ending on a wall raises ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 1), (0, 0))

def test_weighted_optimal_path():
    """Test that A* chooses the lowest-cost path over the shortest-step path."""
    grid = [
        [1, 10, 1],
        [1,  0, 1],
        [1,  1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is not None
    # Direct path through (0,1) costs 10, should be avoided
    assert (0, 1) not in path
    
    # Verify total traversal cost matches the cheaper route
    cost = sum(grid[r][c] for r, c in path[1:])
    assert cost == 6  # 1+1+1+1+1+1 via bottom route