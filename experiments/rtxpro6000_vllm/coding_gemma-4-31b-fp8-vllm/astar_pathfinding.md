```python
import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid.
        :param grid: 2D list where 0 is a wall and >0 is the cost to enter the cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _get_manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        Returns a list of coordinates or None if no path exists.
        """
        # 1. Bounds checking
        for r, c in [start, end]:
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError("Start or end coordinates are out of bounds.")

        # 2. Wall checking
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # 3. Start equals end
        if start == end:
            return [start]

        # Priority Queue: (f_score, current_node)
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # Tracking costs and paths
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current:
                    path.append(current)
                    current = came_from.get(current)
                return path[::-1]

            # 4-directional movement
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols:
                    cost = self.grid[neighbor[0]][neighbor[1]]
                    if cost == 0:  # Wall
                        continue
                    
                    tentative_g_score = g_score[current] + cost
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._get_manhattan_distance(neighbor, end)
                        heapq.heappush(open_set, (f_score, neighbor))

        return None

# --- Pytest Tests ---
import pytest

def calculate_path_cost(grid, path):
    # Start cell cost is usually not counted in pathfinding (cost to enter), 
    # but for consistency we sum all cells in path except the start.
    return sum(grid[r][c] for r, c in path[1:])

def test_simple_path():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5  # 0,0 -> 0,1 -> 0,2 -> 1,2 -> 2,2 (or similar)
    assert calculate_path_cost(grid, path) == 4

def test_obstacles():
    # Wall at (1, 0) and (1, 1) forcing path to go around via (1, 2)
    grid = [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 0))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]

def test_weighted_grid():
    # Path A: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) cost: 1+1+1+1 = 4
    # Path B: (0,0)->(1,0)->(2,0)->(2,1)->(2,2) cost: 10+10+1+1 = 22
    grid = [
        [1, 1, 1],
        [10, 1, 1],
        [10, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
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