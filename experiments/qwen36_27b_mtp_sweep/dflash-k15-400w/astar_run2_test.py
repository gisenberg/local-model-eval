import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid cells with value 0 are walls (impassable).
    Cells with value > 0 are walkable; the value represents the cost to enter that cell.
    Movement is restricted to 4 directions (up, down, left, right).
    Uses Manhattan distance as the admissible and consistent heuristic.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid.
        
        Args:
            grid: 2D list of integers. 0 = wall, >0 = movement cost.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within bounds and not a wall."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] > 0

    def _heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.
        
        Args:
            start: (row, col) tuple for starting position.
            end: (row, col) tuple for target position.
            
        Returns:
            List of (row, col) tuples representing the optimal path, or None if unreachable.
            
        Raises:
            ValueError: If start/end is out of bounds or located on a wall.
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

        # Trivial case
        if start == end:
            return [start]

        # Priority queue: (f_score, counter, position)
        # Counter ensures stable sorting when f_scores are equal
        counter = 0
        open_set: List[Tuple[int, int, Tuple[int, int]]] = [(0, counter, start)]
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            _, _, current = heapq.heappop(open_set)

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
                if not self._is_valid(neighbor):
                    continue

                # Cost to move is the weight of the destination cell
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return None

import pytest

def test_basic_path_avoids_walls():
    """Test standard pathfinding that correctly routes around a wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert (1, 1) not in path  # Must avoid center wall

def test_start_equals_end():
    """Test trivial case where start and end are the same."""
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_no_path_returns_none():
    """Test that unreachable targets return None."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds_start_raises_value_error():
    """Test that out-of-bounds start position raises ValueError."""
    grid = [[1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (0, 0))

def test_wall_at_start_raises_value_error():
    """Test that starting on a wall raises ValueError."""
    grid = [[0]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (0, 0))

def test_weighted_optimal_path():
    """Test that A* chooses the lowest-cost path over the shortest Manhattan path."""
    # High weights on top and bottom rows force path through center
    grid = [
        [1, 100, 1],
        [1, 1, 1],
        [1, 100, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert (1, 1) in path  # Must go through cheap center
    assert (0, 1) not in path  # Avoid expensive top row
    assert (2, 1) not in path  # Avoid expensive bottom row