import heapq
, Tuple, Optional, Dict

class AStarGrid:
    """
    A* Pathfinding implementation on a weighted 2D grid.
    Grid values: 0 = Wall, >0 = Weight/Cost to enter the cell.
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
        # Bounds and validity checks
        for pt in [start, end]:
            if not (0 <= pt[0] < self.rows and 0 <= pt[1] < self.cols):
                raise ValueError("Coordinates out of bounds")
        
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None # Start or end is a wall

        if start == end:
            return [start]

        # Priority Queue stores: (priority, current_cost, current_node)
        # priority = current_cost + heuristic
        open_set = []
        heapq.heappush(open_set, (0, 0, start))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}

        while open_set:
            _, current_cost, current = heapq.heappop(open_set)

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols:
                    weight = self.grid[neighbor[0]][neighbor[1]]
                    if weight == 0: # Wall
                        continue
                    
                    tentative_g_score = current_cost + weight
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._get_heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

        return None

# --- Pytest Tests ---
import pytest

def test_start_is_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_simple_path():
    # 1 = cost 1, 0 = wall
    grid = [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Must go around the wall
    expected = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]
    assert astar.find_path((0, 0), (2, 0)) == expected

def test_weighted_path():
    # Path A: (0,0)->(0,1)->(0,2)->(1,2) | Total cost: 1+1+1 = 3
    # Path B: (0,0)->(1,0)->(1,1)->(1,2) | Total cost: 1+10+1 = 12
    grid = [
        [1, 1, 1],
        [1, 10, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (1, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2)]

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

def test_wall_start_end():
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    # Start is wall
    assert astar.find_path((0, 0), (1, 1)) is None