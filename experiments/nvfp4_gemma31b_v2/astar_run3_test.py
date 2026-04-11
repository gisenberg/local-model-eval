import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    Implements A* pathfinding on a 2D grid.
    Grid values: 0 represents a wall, >0 represents the cost to enter that cell.
    """
    def __init__(self, grid: List[List[int]]):
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _get_heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using A*.
        Returns a list of coordinates or None if no path exists.
        """
        # Out-of-bounds check
        for point in [start, end]:
            if not (0 <= point[0] < self.rows and 0 <= point[1] < self.cols):
                raise ValueError("Start or end point is out of bounds")

        if start == end:
            return [start]

        # Priority Queue: (priority, current_node)
        # priority = g_score + heuristic
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            r, c = current
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (r + dr, c + dc)
                nr, nc = neighbor

                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    weight = self.grid[nr][nc]
                    if weight == 0: # Wall
                        continue
                    
                    tentative_g_score = g_score[current] + weight
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._get_heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score, neighbor))

        return None

# --- Pytest Tests ---
import pytest

def test_start_is_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_simple_path():
    # 1s are cheap, 0s are walls
    grid = [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Must go around the wall
    path = astar.find_path((0, 0), (2, 0))
    expected = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]
    assert path == expected

def test_weighted_path():
    # Path A: (0,0)->(0,1)->(0,2) cost: 1+1 = 2
    # Path B: (0,0)->(1,0)->(1,1)->(1,2)->(0,2) cost: 1+1+1+1 = 4
    # But if we make Path A expensive:
    grid = [
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    # Should take the longer route (bottom) because it's cheaper (1+1+1+1 < 1+10)
    assert path == [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]

def test_no_path():
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) is None

def test_out_of_bounds():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))

def test_complex_maze():
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (4, 0))
    # Path must snake through the maze
    assert path[0] == (0, 0)
    assert path[-1] == (4, 0)
    assert len(path) == 13 # Manhattan distance + detours