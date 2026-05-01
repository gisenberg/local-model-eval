import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.
    
    - Cells with value 0 represent walls (impassable).
    - Positive integers represent traversal costs to enter that cell.
    - Uses 4-directional movement and Manhattan distance heuristic.
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid for pathfinding.
        
        :param grid: 2D list of integers where 0 = wall, >0 = traversal cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic (consistent for 4-directional movement)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using the A* algorithm.
        
        :param start: (row, col) tuple indicating the starting position.
        :param end: (row, col) tuple indicating the target position.
        :return: List of (row, col) tuples representing the optimal path, or None if unreachable.
        :raises ValueError: If start or end is out of bounds or positioned on a wall.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds.")
            
        # Validate walls
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall.")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall.")

        # Handle start == end
        if start == end:
            return [start]

        # A* initialization
        # Priority queue stores (f_score, counter, position)
        open_set: list = []
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))
        
        g_score: dict = {start: 0}
        came_from: dict = {}
        counter = 1  # Tie-breaker to prevent tuple comparison errors in heapq

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
                neighbor = (current[0] + dr, current[1] + dc)
                nr, nc = neighbor

                # Check bounds
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                # Check walls
                if self.grid[nr][nc] == 0:
                    continue

                weight = self.grid[nr][nc]
                tentative_g = g_score[current] + weight

                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1

        return None


# ========================
# PYTEST TESTS
# ========================
import pytest

def test_simple_path():
    """Test basic pathfinding on an unweighted grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Manhattan distance + 1

def test_weighted_optimality():
    """Test that A* chooses the lowest-cost path, not necessarily shortest steps."""
    grid = [
        [1, 1, 1, 1],
        [1, 10, 10, 1],
        [1, 10, 10, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (3, 3))
    assert path is not None
    # Cost = sum of weights of entered cells (excluding start)
    cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert cost == 6  # Optimal path skirts the expensive center
    assert (1, 1) not in path  # Should avoid high-cost cells

def test_no_path_exists():
    """Test that None is returned when the target is unreachable."""
    grid = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) is None

def test_start_equals_end():
    """Test handling of identical start and end positions."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((1, 1), (1, 1)) == [(1, 1)]

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((2, 2), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (-1, 0))

def test_wall_at_start_or_end_raises_value_error():
    """Test that starting or ending on a wall raises ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((1, 1), (0, 0))