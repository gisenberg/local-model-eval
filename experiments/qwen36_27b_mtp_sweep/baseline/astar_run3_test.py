import heapq
import math
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.
    
    Grid representation:
    - 0: Wall (impassable)
    - >0: Traversal cost/weight to enter the cell
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid for A* pathfinding.
        
        :param grid: 2D list of integers. 0 represents walls, positive integers represent weights.
        :raises ValueError: If grid is empty or rows have inconsistent lengths.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        if any(len(row) != len(grid[0]) for row in grid):
            raise ValueError("All rows must have the same length")
            
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self._directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic (admissible & consistent for 4-dir movement with weights >= 1)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using the A* algorithm.
        
        :param start: (row, col) tuple for the starting position.
        :param end: (row, col) tuple for the target position.
        :return: List of (row, col) tuples representing the optimal path, or None if unreachable.
        :raises ValueError: If start or end is out of bounds or positioned on a wall.
        """
        # Validate start and end positions
        for pos, name in [(start, "start"), (end, "end")]:
            r, c = pos
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(f"{name} is out of bounds")
            if self.grid[r][c] == 0:
                raise ValueError(f"{name} is on a wall")

        if start == end:
            return [start]

        # Priority queue stores tuples: (f_score, tie_breaker, position)
        open_set: List[Tuple[float, int, Tuple[int, int]]] = [(0, 0, start)]
        tie_breaker = 0
        
        # g_score: minimum known cost from start to node
        g_score: dict[Tuple[int, int], float] = {start: 0}
        # came_from: tracks parent nodes for path reconstruction
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path by backtracking
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            cr, cc = current
            for dr, dc in self._directions:
                nr, nc = cr + dr, cc + dc
                neighbor = (nr, nc)

                # Skip out-of-bounds or walls
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr][nc] == 0:
                    continue

                # Cost to enter the neighbor cell
                move_cost = self.grid[nr][nc]
                tentative_g = g_score[current] + move_cost

                # Update if we found a cheaper path to neighbor
                if tentative_g < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    tie_breaker += 1
                    heapq.heappush(open_set, (f_score, tie_breaker, neighbor))

        return None

import pytest
from typing import List, Tuple, Optional

# Assuming AStarGrid is imported or defined in the same file
# 
def test_basic_pathfinding():
    """Test simple path on an unweighted grid."""
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
    # Manhattan distance + 1 (start node) = 5 steps
    assert len(path) == 5

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_exists():
    """Test when end is completely blocked by walls."""
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
    with pytest.raises(ValueError, match="start is out of bounds"):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError, match="end is out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_position_raises_value_error():
    """Test that placing start/end on a wall raises ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="start is on a wall"):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError, match="end is on a wall"):
        astar.find_path((1, 1), (0, 0))

def test_weighted_optimal_path():
    """Test that A* correctly chooses the lower-cost path over the shorter geometric path."""
    # Top row has high weights, bottom rows have low weights
    grid = [
        [1, 10, 10, 1],
        [1,  1,  1, 1],
        [1,  1,  1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 3))
    
    assert path is not None
    # Optimal path should detour through row 1 to avoid weight-10 cells
    expected = [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (0, 3)]
    assert path == expected
    
    # Verify cost calculation: sum of weights of all cells except start
    path_cost = sum(grid[r][c] for r, c in path[1:])
    assert path_cost == 5  # 1+1+1+1+1