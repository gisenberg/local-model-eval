import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    Implements A* pathfinding on a 2D weighted grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the grid. grid[r][c] is the cost to enter that cell. 
        0 represents an impassable wall.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, r: int, c: int) -> bool:
        """Checks if a coordinate is within bounds and not a wall."""
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] > 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculates Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using A*.
        Returns a list of (row, col) tuples or None if no path exists.
        """
        # Bounds and Wall checks
        for point in [start, end]:
            if not (0 <= point[0] < self.rows and 0 <= point[1] < self.cols):
                raise ValueError(f"Coordinate {point} is out of bounds.")
        
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        if start == end:
            return [start]

        # Priority Queue stores (f_score, current_node)
        # g_score stores the cost to reach the node
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        g_score = {start: 0}
        came_from = {}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            r, c = current
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (r + dr, c + dc)

                if self._is_valid(neighbor[0], neighbor[1]):
                    # Cost to enter neighbor is the value in the grid
                    tentative_g_score = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score, neighbor))

        return None

# --- Pytest Tests ---
import pytest

def calculate_path_cost(grid, path):
    """Helper to calculate total cost of a path (excluding start cell cost)."""
    if not path: return 0
    return sum(grid[r][c] for r, c in path[1:])

def test_simple_path():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert calculate_path_cost(grid, path) == 4

def test_path_around_obstacles():
    # 0 is a wall
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 0), (1, 2))
    assert path is not None
    # Must go around the center wall
    assert (1, 1) not in path
    assert calculate_path_cost(grid, path) == 4

def test_weighted_grid():
    # Path through (0,1) is expensive, path through (1,0) is cheap
    grid = [
        [1, 10, 1],
        [1, 1,  1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    # Optimal path should be (0,0) -> (1,0) -> (1,1) -> (1,2) -> (0,2)
    # Total cost: 1 + 1 + 1 + 1 = 4 (vs 10 + 1 = 11)
    assert path == [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]
    assert calculate_path_cost(grid, path) == 4

def test_no_path_exists():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_start_equals_end():
    grid = [[5]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_invalid_coordinates():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))

if __name__ == "__main__":
    # This allows running tests via 'python filename.py' if pytest is installed
    pytest.main([__file__])