import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.

    The grid is represented as a list of lists of numbers where:
    - 0 represents a wall (impassable)
    - Positive numbers represent the traversal cost/weight of a cell
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the grid for pathfinding."""
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start: (row, col) tuple of the starting position
            end: (row, col) tuple of the target position

        Returns:
            A list of (row, col) tuples representing the optimal path,
            or None if no path exists.

        Raises:
            ValueError: If start or end is out of bounds or positioned on a wall.
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

        # Priority queue: (f_score, tie_breaker, (row, col))
        pq: List[Tuple[float, int, Tuple[int, int]]] = [(0, 0, start)]
        counter = 0

        g_score: Dict[Tuple[int, int], float] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while pq:
            _, _, current = heapq.heappop(pq)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # 4-directional movement
            for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nr, nc = current[0] + dr, current[1] + dc
                
                # Check bounds and walls
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != 0:
                    tentative_g = g_score[current] + self.grid[nr][nc]
                    
                    if tentative_g < g_score.get((nr, nc), float('inf')):
                        came_from[(nr, nc)] = current
                        g_score[(nr, nc)] = tentative_g
                        
                        # Manhattan heuristic
                        h_score = abs(nr - end[0]) + abs(nc - end[1])
                        f_score = tentative_g + h_score
                        
                        counter += 1
                        heapq.heappush(pq, (f_score, counter, (nr, nc)))

        return None

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
    # Optimal path length in steps should be 5 (Manhattan distance + 1)
    assert len(path) == 5

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_no_path_exists():
    """Test when walls completely block the route."""
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
    
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (0, 0))
        
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_at_start_or_end_raises_value_error():
    """Test that placing start/end on a wall raises ValueError."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))
        
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((1, 1), (0, 0))

def test_weighted_optimal_path():
    """Test that A* chooses the lowest-cost path, not just shortest steps."""
    # Top row has high weights, bottom row has low weights
    grid = [
        [1, 10, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is not None
    # Calculate actual traversal cost (excluding start cell)
    cost = sum(astar.grid[r][c] for r, c in path[1:])
    
    # Direct top route costs 10 + 1 = 11
    # Bottom route costs 1 + 1 + 1 + 1 = 4
    assert cost == 4
    assert (1, 0) in path  # Verifies it detours through the cheaper row