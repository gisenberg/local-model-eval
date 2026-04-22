import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    The grid is represented as a 2D list where 0 indicates a wall 
    and positive integers indicate the cost to enter that cell.
    Movement is restricted to 4 directions (up, down, left, right).
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the AStarGrid with a 2D grid.
        
        :param grid: 2D list of integers. 0 = wall, >0 = cell weight.
        :raises ValueError: If grid is empty.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* search.
        
        :param start: (row, col) tuple for the starting position.
        :param end: (row, col) tuple for the target position.
        :return: List of (row, col) tuples representing the optimal path,
                 or None if no valid path exists.
        :raises ValueError: If start/end are out of bounds or located on a wall.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position out of bounds")

        # Validate walls
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            raise ValueError("Start or end position is a wall")

        if start == end:
            return [start]

        # Priority queue: (f_score, counter, (row, col))
        # counter ensures stable ordering when f_scores are equal
        open_set = []
        heapq.heappush(open_set, (0, 0, start))

        # g_score: cost from start to current cell
        g_score = {start: 0}
        # came_from: tracks the path for reconstruction
        came_from = {}
        counter = 1

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

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

            for dr, dc in directions:
                nr, nc = current[0] + dr, current[1] + dc

                # Check bounds and walls
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != 0:
                    # Cost to enter the neighbor cell
                    tentative_g = g_score[current] + self.grid[nr][nc]

                    if tentative_g < g_score.get((nr, nc), float('inf')):
                        came_from[(nr, nc)] = current
                        g_score[(nr, nc)] = tentative_g
                        # Manhattan heuristic
                        f_score = tentative_g + abs(nr - end[0]) + abs(nc - end[1])
                        heapq.heappush(open_set, (f_score, counter, (nr, nc)))
                        counter += 1

        return None

import pytest
# 
def test_basic_path():
    """Test standard pathfinding on an open grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Manhattan distance + 1

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((1, 1), (1, 1)) == [(1, 1)]

def test_no_path_exists():
    """Test that None is returned when the target is unreachable."""
    grid = [[1, 0, 1], [1, 0, 1], [1, 1, 1]]
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

def test_wall_at_start_or_end_raises_value_error():
    """Test that starting or ending on a wall raises ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((1, 1), (0, 0))

def test_weighted_optimal_path():
    """Test that A* correctly chooses a cheaper detour over a direct high-cost path."""
    # Direct path costs 100+1=101, detour costs 1+1+1+1=4
    grid = [[1, 100, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path == [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]