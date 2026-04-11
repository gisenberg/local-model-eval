import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* Pathfinding implementation on a weighted 2D grid.
    Grid values: 0 = Wall, >0 = Cost to enter that cell.
    """
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _get_manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using A*.
        Returns a list of coordinates or None if no path exists.
        """
        # Validation
        for pt in [start, end]:
            if not (0 <= pt[0] < self.rows and 0 <= pt[1] < self.cols):
                raise ValueError("Coordinates out of bounds")

        if start == end:
            return [start]

        # Priority Queue: (f_score, current_node)
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
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

            # 4-directional movement
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)

                # Bounds and Wall check
                if (0 <= neighbor[0] < self.rows and 
                    0 <= neighbor[1] < self.cols and 
                    self.grid[neighbor[0]][neighbor[1]] != 0):
                    
                    # Weight of the cell we are moving into
                    weight = self.grid[neighbor[0]][neighbor[1]]
                    tentative_g_score = g_score[current] + weight

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._get_manhattan_distance(neighbor, end)
                        heapq.heappush(open_set, (f_score, neighbor))

        return None

import pytest

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_simple_path():
    # 1 = normal cost, 0 = wall
    grid = [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    expected = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]
    assert astar.find_path((0, 0), (2, 0)) == expected

def test_weighted_path_optimality():
    # Path A (top) is shorter distance but higher cost (10)
    # Path B (bottom) is longer distance but lower cost (1)
    grid = [
        [1, 10, 1],
        [1, 1,  1],
        [1, 1,  1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    # Should avoid the '10' cell and go around via (1,0)->(1,1)->(1,2)->(0,2)
    assert (0, 1) not in path

def test_no_path_possible():
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
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 0, 1],
        [0, 0, 1, 0, 1],
        [1, 1, 1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (4, 4))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)