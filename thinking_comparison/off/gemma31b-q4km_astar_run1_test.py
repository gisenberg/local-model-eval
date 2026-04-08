import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid.
        grid: 2D list where 0 = wall, positive int = cost to enter cell.
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
        # Bounds checking
        for pt in [start, end]:
            if not (0 <= pt[0] < self.rows and 0 <= pt[1] < self.cols):
                raise ValueError("Start or end coordinates are out of bounds.")

        # Wall checking
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Edge case: start is end
        if start == end:
            return [start]

        # Priority Queue: (priority, current_cost, current_node)
        # priority = current_cost + heuristic
        open_set = []
        heapq.heappush(open_set, (0 + self._get_manhattan_distance(start, end), 0, start))
        
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        g_score: Dict[Tuple[int, int], int] = {start: 0}

        while open_set:
            _, current_cost, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            # 4-directional movement
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dr, current[1] + dc)

                # Check bounds and if it's a wall
                if (0 <= neighbor[0] < self.rows and 
                    0 <= neighbor[1] < self.cols and 
                    self.grid[neighbor[0]][neighbor[1]] != 0):
                    
                    # Cost to enter the neighbor cell
                    tentative_g_score = current_cost + self.grid[neighbor[0]][neighbor[1]]

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._get_manhattan_distance(neighbor, end)
                        heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

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
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5 # (0,0)->(0,1)->(0,2)->(1,2)->(2,2) or similar
    assert calculate_path_cost(grid, path) == 4

def test_path_around_obstacles():
    # 0 is wall
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]

def test_weighted_grid_optimality():
    # Path through (1,1) is shorter distance but higher cost
    # Path through (0,1) is longer distance but lower cost
    grid = [
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    # Optimal path should avoid the '10' cell
    # Expected path: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) cost: 1+1+1+1 = 4
    # Direct path: (0,0) -> (1,0) -> (1,1) -> (1,2) -> (2,2) cost: 1+10+1+1 = 13
    assert calculate_path_cost(grid, path) == 4
    assert (1, 1) not in path

def test_no_path_exists():
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