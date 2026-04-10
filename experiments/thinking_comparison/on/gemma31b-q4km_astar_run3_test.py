import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    Implements A* pathfinding on a 2D grid where cells have movement costs.
    0 represents an impassable wall, and positive integers represent the cost to enter that cell.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D list of costs.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate the Manhattan distance between two points.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Return valid 4-directional neighbors that are not walls.
        """
        row, col = node
        neighbors = []
        # Up, Down, Left, Right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            if 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != 0:
                neighbors.append((r, c))
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using the A* algorithm.
        Returns a list of coordinates or None if no path exists.
        """
        # 1. Bounds check
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols) or \
           not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("Start or end coordinates are out of grid bounds.")

        # 2. Wall check
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # 3. Start == End check
        if start == end:
            return [start]

        # Priority Queue stores (f_score, current_node)
        open_set = []
        heapq.heappush(open_set, (0 + self._heuristic(start, end), start))
        
        # came_from maps a node to its predecessor for path reconstruction
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score: cost from start to current node
        g_score: Dict[Tuple[int, int], float] = {start: 0}

        while open_set:
            # Pop node with the lowest f_score
            current_f, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in self._get_neighbors(current):
                # Cost to enter the neighbor cell
                tentative_g_score = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))

        return None

# --- Pytest Tests ---
import pytest

def calculate_path_cost(grid, path):
    """Helper to calculate total cost of a path (excluding start cell)."""
    if not path: return 0
    cost = 0
    for i in range(1, len(path)):
        r, c = path[i]
        cost += grid[r][c]
    return cost

def test_simple_uniform_grid():
    grid = [[1, 1, 1], 
            [1, 1, 1], 
            [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5  # (0,0)->(0,1)->(0,2)->(1,2)->(2,2) or similar
    assert calculate_path_cost(grid, path) == 4

def test_path_around_obstacles():
    # 0 is wall
    grid = [[1, 0, 1], 
            [1, 0, 1], 
            [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    # Must go down to (2,0), across to (2,2), then up to (0,2)
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]
    assert calculate_path_cost(grid, path) == 6

def test_weighted_grid_optimality():
    # Path A: (0,0) -> (0,1) -> (0,2) [Cost: 10+10 = 20]
    # Path B: (0,0) -> (1,0) -> (1,1) -> (1,2) -> (0,2) [Cost: 1+1+1+1 = 4]
    grid = [[1, 10, 10], 
            [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path == [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]
    assert calculate_path_cost(grid, path) == 4

def test_no_path_exists():
    grid = [[1, 0, 1], 
            [0, 0, 0], 
            [1, 0, 1]]
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

if __name__ == "__main__":
    # To run tests manually without pytest CLI:
    pytest.main([__file__])