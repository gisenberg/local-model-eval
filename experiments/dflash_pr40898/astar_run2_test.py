import heapq
from typing import List, Tuple, Optional, Dict


class AStarGrid:
    """A* pathfinding algorithm on a weighted 2D grid.
    
    Cells with value 0 are treated as walls. Positive integers represent 
    the cost to enter that cell.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid.
        
        Args:
            grid: 2D list where 0 represents a wall and positive integers 
                  represent the traversal cost to enter the cell.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within bounds and not a wall."""
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic for 4-directional movement."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: Tuple (row, col) of the starting cell.
            end: Tuple (row, col) of the target cell.
            
        Returns:
            List of (row, col) tuples representing the optimal path, 
            or None if no path exists.
            
        Raises:
            ValueError: If start/end are out of bounds or located on a wall.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds")
            
        # Validate walls
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            raise ValueError("Start or end position is a wall")

        # Handle start == end
        if start == end:
            return [start]

        # Priority queue: (f_score, tie_breaker_counter, (row, col))
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))

        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        counter = 1

        while open_set:
            f, _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Explore 4-directional neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = current[0] + dr, current[1] + dc
                if not self._is_valid(nr, nc):
                    continue

                # Cost to enter neighbor cell
                tentative_g = g_score[current] + self.grid[nr][nc]

                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + self._heuristic((nr, nc), end)
                    heapq.heappush(open_set, (f_score, counter, (nr, nc)))
                    counter += 1

        return None

import pytest

def test_basic_pathfinding():
    """Test standard path around a wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_unreachable_path():
    """Test when end is completely blocked by walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds_start():
    """Test ValueError when start is outside grid dimensions."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))

def test_wall_at_start():
    """Test ValueError when start or end is on a wall."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))

def test_weighted_optimality():
    """Test that A* chooses the lower-cost path over a shorter geometric path."""
    # Top path cost: 10 + 10 + 1 = 21
    # Bottom path cost: 1 + 1 + 1 + 1 = 4
    grid = [
        [1, 10, 10, 1],
        [1,  1,  1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 3))
    # Should route through row 1 to minimize total weight
    assert path == [(0, 0), (1, 0), (1, 1), (1, 2), (1, 3), (0, 3)]