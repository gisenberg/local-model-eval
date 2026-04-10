import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    Implements A* pathfinding on a 2D grid where cells have movement costs.
    0 represents an impassable wall, and positive integers represent the cost to enter the cell.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D list of costs.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate the Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Return valid 4-directional neighbors that are not walls."""
        r, c = node
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != 0:
                neighbors.append((nr, nc))
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using the A* algorithm.
        Returns a list of coordinates or None if no path exists.
        """
        # 1. Bounds Check
        for point in [start, end]:
            if not (0 <= point[0] < self.rows and 0 <= point[1] < self.cols):
                raise ValueError("Start or end coordinates are out of bounds.")

        # 2. Edge Case: Start or End is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # 3. Edge Case: Start equals End
        if start == end:
            return [start]

        # Priority Queue: (f_score, current_node)
        open_set = []
        heapq.heappush(open_set, (0, start))

        # Track the path and the cost to reach each node
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        
        came_from[start] = None

        while open_set:
            # Pop node with the lowest f_score
            _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for neighbor in self._get_neighbors(current):
                # Cost to enter the neighbor cell
                tentative_g_score = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
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
    cost = 0
    for i in range(1, len(path)):
        r, c = path[i]
        cost += grid[r][c]
    return cost

def test_simple_uniform_grid():
    # 1s everywhere, simple L-shape or straight line
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5  # (0,0)->(0,1)->(0,2)->(1,2)->(2,2) or similar
    assert calculate_path_cost(grid, path) == 4

def test_path_around_obstacles():
    # Wall blocking the direct path
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 0), (1, 2))
    # Must go around the center wall: (1,0) -> (0,0) -> (0,1) -> (0,2) -> (1,2)
    assert path is not None
    assert (1, 1) not in path
    assert len(path) == 5

def test_weighted_grid_optimality():
    # Path A: Direct but expensive (cost 10)
    # Path B: Longer but cheaper (cost 1+1+1+1)
    grid = [
        [1, 1, 1, 1],
        [1, 10, 10, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Start (1,0) to End (1,3)
    # Direct path (1,0)->(1,1)->(1,2)->(1,3) cost: 10+10+1 = 21
    # Detour path (1,0)->(0,0)->(0,1)->(0,2)->(0,3)->(1,3) cost: 1+1+1+1+1 = 5
    path = astar.find_path((1, 0), (1, 3))
    assert path is not None
    assert calculate_path_cost(grid, path) == 5
    assert (1, 1) not in path

def test_no_path_exists():
    # End is completely boxed in by walls (0s)
    grid = [
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 1] # (2,2) is isolated
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_invalid_coordinates():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))

if __name__ == "__main__":
    # To run tests without pytest installed, you can use:
    # pytest this_file.py
    print("Run 'pytest <filename>.py' to execute tests.")