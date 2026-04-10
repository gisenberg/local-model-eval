import heapq
, Tuple, Optional, Dict

class AStarGrid:
    """
    Implements A* pathfinding on a weighted 2D grid.
    Grid values: 0 = Wall, >0 = Cost to enter the cell.
    """
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _get_manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using A*.
        Returns a list of coordinates including start and end, or None if no path exists.
        """
        # Bounds and validity checks
        for pt in [start, end]:
            if not (0 <= pt[0] < self.rows and 0 <= pt[1] < self.cols):
                raise ValueError("Start or end position is out of bounds.")
        
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None # Start or end is a wall

        if start == end:
            return [start]

        # Priority Queue stores: (f_score, current_node)
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        # g_score: cost from start to node
        g_score: Dict[Tuple[int, int], int] = {start: 0}

        while open_set:
            _, current = heapq.heappop(open_set)

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
                    
                    tentative_g_score = g_score[current] + weight
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._get_manhattan_distance(neighbor, end)
                        heapq.heappush(open_set, (f_score, neighbor))

        return None

# --- Pytest Tests ---
import pytest

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_simple_path():
    # 1 is low cost, 10 is high cost
    grid = [
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Should go around the 10
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert (1, 1) not in path

def test_wall_blocking():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None

def test_out_of_bounds():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))

def test_weighted_optimality():
    # Path A: (0,0)->(0,1)->(0,2) cost: 1+1 = 2
    # Path B: (0,0)->(1,0)->(1,1)->(1,2)->(0,2) cost: 1+1+1+1 = 4
    grid = [
        [1, 1, 1],
        [1, 1, 1]
    ]
    # Make the direct route very expensive
    grid[0][1] = 10 
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    # Should take the long way (bottom row) because it's cheaper
    assert (1, 0) in path and (1, 1) in path and (1, 2) in path

def test_start_or_end_is_wall():
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (1, 1)) is None