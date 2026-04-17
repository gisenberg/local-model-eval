```python
import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    Implements A* pathfinding on a 2D grid where cells have movement costs.
    0 represents an impassable wall, and positive integers represent the cost to enter that cell.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a cost map.
        
        :param grid: A 2D list of integers where 0 is a wall and >0 is the cost to enter.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate the Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from start to end using the A* algorithm.
        
        :param start: (row, col) starting coordinates.
        :param end: (row, col) destination coordinates.
        :return: List of (row, col) coordinates from start to end, or None if no path exists.
        :raises ValueError: If start or end coordinates are outside the grid boundaries.
        """
        # 1. Boundary Checks
        for point in [start, end]:
            r, c = point
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(f"Coordinates {point} are out of grid bounds.")

        # 2. Edge Cases: Start == End or Wall obstacles
        if start == end:
            return [start]
        
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Priority Queue: (f_score, (row, col))
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, end), start))
        
        # Tracking costs and paths
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        
        # To avoid processing the same node multiple times with a higher cost
        visited = set()

        while open_set:
            # Get node with lowest f_score (g_score + heuristic)
            current_f, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            if current in visited:
                continue
            visited.add(current)

            r, c = current
            # 4-directional movement
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (r + dr, c + dc)
                nr, nc = neighbor

                # Check bounds and if the cell is a wall
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    cost = self.grid[nr][nc]
                    if cost == 0:
                        continue
                    
                    # Tentative g_score is the cost to reach the neighbor
                    tentative_g_score = g_score[current] + cost
                    
                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score, neighbor))

        return None

# ==========================================
# Pytest Tests
# ==========================================
import pytest

def calculate_path_cost(grid, path):
    """Helper to calculate total cost of a path (excluding start cell)."""
    if not path: return float('inf')
    return sum(grid[r][c] for r, c in path[1:])

def test_simple_uniform_grid():
    # 3x3 grid, all costs = 1
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is not None
    assert path[0] == start and path[-1] == end
    assert calculate_path_cost(grid, path) == 4  # Manhattan distance

def test_path_around_obstacles():
    # Wall blocking the direct path
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start, end = (1, 0), (1, 2)
    path = astar.find_path(start, end)
    # Must go around the center wall (1,0) -> (0,0) -> (0,1) -> (0,2) -> (1,2)
    assert path is not None
    assert (1, 1) not in path
    assert calculate_path_cost(grid, path) == 4

def test_weighted_grid_optimality():
    # Path A (direct) is expensive, Path B (long way) is cheap
    # S 1 1
    # 10 10 1
    # 1 1 1  <- End is at (2,2)
    grid = [
        [1, 1, 1],
        [10, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    # Optimal path: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) | Cost: 1+1+1+1 = 4
    # Direct path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2) | Cost: 10+1+1+1 = 13
    assert calculate_path_cost(grid, path) == 4
    assert (1, 0) not in path

def test_no_path_exists():
    # End is completely boxed in by walls
    grid = [
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is None

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    start = (0, 0)
    path = astar.find_path(start, start)
    assert path == [start]

def test_invalid_coordinates():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))
```