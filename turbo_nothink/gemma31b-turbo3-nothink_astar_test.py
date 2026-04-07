import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    def __init__(self, grid: List[List[int]]:
        """
        Initialize the A* pathfinder with a 2D grid.
        grid[r][c] = 0 means impassable, > 0 is the cost to enter that cell.
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
        # Bounds check
        for p in [start, end]:
            if not (0 <= p[0] < self.rows and 0 <= p[1] < self.cols):
                raise ValueError("Start or end coordinates are out of bounds.")

        # Wall check
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

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
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            r, c = current
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (r + dr, c + dc)
                nr, nc = neighbor

                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    cost = self.grid[nr][nc]
                    if cost == 0: # Wall
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
    """Helper to calculate total cost of a path (excluding start cell)."""
    cost = 0
    for i in range(1, len(path)):
        r, c = path[i]
        cost += grid[r][c]
    return cost

def test_simple_uniform():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)] or \
           path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)] # Both are optimal
    assert calculate_path_cost(grid, path) == 4

def test_obstacles():
    grid = [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 0))
    # Must go around the wall
    expected = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]
    assert path == expected
    assert calculate_path_cost(grid, path) == 6

def test_weighted_grid():
    # Path (0,0)->(0,1)->(0,2) costs 1+1=2
    # Path (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2) costs 1+1+1+1+1+1=6
    grid = [
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path == [(0, 0), (0, 1), (0, 2)]
    assert calculate_path_cost(grid, path) == 2

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

def test_invalid_coords():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))

if __name__ == "__main__":
    # To run tests without pytest installed, you can use: pytest filename.py
    pytest.main([__file__])