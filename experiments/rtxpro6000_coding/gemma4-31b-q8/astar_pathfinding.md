```python
import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* pathfinder on a weighted 2D grid.
        
        :param grid: 2D list where 0 is a wall and positive integers are movement costs.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _get_manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate the Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using the A* algorithm.
        
        :param start: (row, col) starting coordinates.
        :param end: (row, col) ending coordinates.
        :return: List of coordinates from start to end, or None if no path exists.
        :raises ValueError: If start or end coordinates are out of grid bounds.
        """
        # 1. Bounds checking
        for r, c in [start, end]:
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError("Start or end coordinates are out of bounds.")

        # 2. Edge cases: start/end are walls or start equals end
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        if start == end:
            return [start]

        # Priority Queue: (f_score, current_node)
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # Tracking costs and paths
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        
        # f_score[n] = g_score[n] + h(n)
        f_score: Dict[Tuple[int, int], float] = {start: self._get_manhattan_distance(start, end)}

        while open_set:
            # Get node with the lowest estimated total cost
            _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Explore 4-directional neighbors
            r, c = current
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (r + dr, c + dc)
                nr, nc = neighbor

                # Check bounds and if the cell is a wall
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    cost = self.grid[nr][nc]
                    if cost == 0:
                        continue
                    
                    # Tentative g_score is distance from start to neighbor through current
                    tentative_g_score = g_score[current] + cost

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        # This path to neighbor is better than any previous one
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._get_manhattan_distance(neighbor, end)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

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

def test_simple_uniform_grid():
    # 3x3 grid, all costs = 1
    grid = [[1]*3 for _ in range(3)]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    assert calculate_path_cost(grid, path) == 4  # 4 steps to get from 0,0 to 2,2

def test_path_around_obstacles():
    # Wall in the middle
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is not None
    assert (1, 1) not in path
    assert calculate_path_cost(grid, path) == 4

def test_weighted_grid_optimality():
    # Path A (direct) is distance 2 but cost 10
    # Path B (detour) is distance 4 but cost 4
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start, end = (0, 0), (0, 2)
    path = astar.find_path(start, end)
    # Optimal path should go (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2) -> (1,2) -> (0,2)
    # Total cost: 1+1+1+1+1+1 = 6. Direct path (0,0)->(0,1)->(0,2) cost: 10+1 = 11.
    assert calculate_path_cost(grid, path) == 6
    assert (0, 1) not in path

def test_no_path_exists():
    # End is completely boxed in by walls
    grid = [
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) is None

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    start = (0, 0)
    assert astar.find_path(start, start) == [start]

def test_invalid_coordinates():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))
```