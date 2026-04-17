```python
import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize AStarGrid with a 2D list of costs.
        0 represents an impassable wall.
        Positive integers represent the cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _get_manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        Returns a list of coordinates or None if no path exists.
        """
        # 1. Bounds checking
        for pt in [start, end]:
            if not (0 <= pt[0] < self.rows and 0 <= pt[1] < self.cols):
                raise ValueError("Start or end coordinates are out of bounds")

        # 2. Wall checking
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # 3. Start == End case
        if start == end:
            return [start]

        # Priority Queue stores: (f_score, (row, col))
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # Tracks the cost to reach a node
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        # Tracks the path
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        while open_set:
            current_f, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            r, c = current
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (r + dr, c + dc)
                nr, nc = neighbor

                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    weight = self.grid[nr][nc]
                    if weight == 0: # Wall
                        continue
                    
                    # Tentative g_score is distance from start to neighbor
                    tentative_g = g_score[current] + weight
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + self._get_manhattan_distance(neighbor, end)
                        heapq.heappush(open_set, (f_score, neighbor))
                        came_from[neighbor] = current

        return None

# --- Pytest Tests ---
import pytest

def calculate_path_cost(grid, path):
    """Helper to calculate total cost of a path (excluding start cell cost)."""
    cost = 0
    for i in range(1, len(path)):
        r, c = path[i]
        cost += grid[r][c]
    return cost

def test_simple_path():
    # Uniform grid (all costs = 1)
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5 # (0,0)->(0,1)->(0,2)->(1,2)->(2,2) or similar
    assert calculate_path_cost(grid, path) == 4

def test_obstacles():
    # Grid with a wall blocking the direct path
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    # Must go around the center wall: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2)
    # Wait, actually (0,0)->(0,1)->(0,2) is blocked by (1,1) but (0,1) is open.
    # Let's use a more restrictive wall.
    grid_blocked = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar_blocked = AStarGrid(grid_blocked)
    path = astar_blocked.find_path((0, 0), (0, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]

def test_weighted_grid():
    # Path prefers lower cost cells even if it's longer
    # (0,0) -> (0,1) cost 10 -> (0,2) cost 10 = 20
    # (0,0) -> (1,0) cost 1 -> (1,1) cost 1 -> (1,2) cost 1 -> (0,2) cost 1 = 4
    grid = [
        [1, 10, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path == [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]
    assert calculate_path_cost(grid, path) == 4

def test_no_path():
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) is None

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_invalid_coordinates():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))
```