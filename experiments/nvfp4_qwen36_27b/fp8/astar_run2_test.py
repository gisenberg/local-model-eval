import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    The grid is represented as a list of lists of numbers.
    - 0 represents a wall (impassable).
    - Positive numbers represent the cost to enter that cell.
    - Movement is restricted to 4 directions (up, down, left, right).
    """

    def __init__(self, grid: List[List[float]]) -> None:
        """
        Initialize the AStarGrid with a 2D grid.
        
        Args:
            grid: 2D list of numbers where 0 is a wall and >0 is the cell weight.
            
        Raises:
            ValueError: If grid is empty or not rectangular.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        if any(len(row) != len(grid[0]) for row in grid):
            raise ValueError("Grid must be rectangular")
            
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic (admissible for 4-directional movement)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.
        
        Args:
            start: Tuple (row, col) of the starting position.
            end: Tuple (row, col) of the target position.
            
        Returns:
            A list of (row, col) tuples representing the optimal path, 
            or None if no valid path exists.
            
        Raises:
            ValueError: If start or end is out of bounds or located on a wall.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds")

        # Validate walls
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall")

        if start == end:
            return [start]

        # Priority queue: (f_score, counter, (row, col))
        # Counter breaks ties to avoid comparing tuples
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))

        g_score: Dict[Tuple[int, int], float] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        counter = 1

        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                return path[::-1]

            for dr, dc in directions:
                nr, nc = current[0] + dr, current[1] + dc

                # Bounds check
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                # Wall check
                if self.grid[nr][nc] == 0:
                    continue

                weight = self.grid[nr][nc]
                tentative_g = g_score[current] + weight

                if tentative_g < g_score.get((nr, nc), float('inf')):
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + self._heuristic((nr, nc), end)
                    heapq.heappush(open_set, (f_score, counter, (nr, nc)))
                    counter += 1
                    came_from[(nr, nc)] = current

        return None

import pytest

def _path_cost(grid, path):
    """Helper to calculate total cost of a path (sum of entered cells)."""
    return sum(grid[r][c] for r, c in path[1:])

def test_basic_pathfinding():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Minimum steps in 3x3 grid

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_exists():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds_raises_value_error():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_positions_raise_value_error():
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((1, 0), (0, 0))

def test_weighted_optimality():
    # High weight in the middle forces A* to take a longer but cheaper route
    grid = [
        [1, 100, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is not None
    assert (0, 1) not in path  # Must avoid the expensive cell
    assert _path_cost(grid, path) == 4  # 1+1+1+1 (enters 4 cells of weight 1)
    assert len(path) == 5  # Longer path, but optimal cost