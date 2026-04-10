import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* Pathfinding implementation on a weighted 2D grid.
    Grid values: 0 = Wall, >0 = Weight (cost to enter cell).
    """
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.
        :param grid: 2D list where 0 represents a wall and positive integers represent cost.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _get_manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using A*.
        Returns a list of coordinates or None if no path exists.
        """
        # Bounds and validity checks
        for pt in [start, end]:
            if not (0 <= pt[0] < self.rows and 0 <= pt[1] < self.cols):
                raise ValueError("Start or end coordinates are out of bounds")
        
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None # Start or end is a wall

        if start == end:
            return [start]

        # Priority Queue: (f_score, current_node)
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # Tracking
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
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

            # 4-directional movement
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols:
                    weight = self.grid[neighbor[0]][neighbor[1]]
                    if weight == 0: # Wall
                        continue
                    
                    # Tentative g_score is current g_score + weight of the cell we are entering
                    tentative_g = g_score[current] + weight
                    
                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + self._get_manhattan_distance(neighbor, end)
                        heapq.heappush(open_set, (f_score, neighbor))

        return None

# --- Pytest Tests ---
import pytest

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_out_of_bounds():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))

def test_no_path_wall():
    # Wall blocking the path
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None

def test_optimal_weighted_path():
    # Path A: (0,0)->(0,1)->(0,2) cost: 10+10 = 20
    # Path B: (0,0)->(1,0)->(1,1)->(1,2)->(0,2) cost: 1+1+1+1 = 4
    grid = [
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    expected = [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]
    assert path == expected

def test_simple_path():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    # Any path of length 5 is optimal here
    assert len(path) == 5 
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

def test_start_or_end_is_wall():
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    # Start is wall
    assert astar.find_path((0, 0), (1, 1)) is None
    # End is wall
    grid2 = [[1, 1], [1, 0]]
    astar2 = AStarGrid(grid2)
    assert astar2.find_path((0, 0), (1, 1)) is None