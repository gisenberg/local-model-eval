import heapq
, Tuple, Optional, Dict

class AStarGrid:
    """
    Implements A* pathfinding on a weighted 2D grid.
    Grid values: 0 represents a wall, >0 represents the cost to enter that cell.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid.
        :param grid: A 2D list of integers where 0 is a wall and >0 is the weight.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _get_manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using A*.
        :return: List of coordinates representing the path, or None if no path exists.
        :raises ValueError: If start or end coordinates are out of bounds.
        """
        # Bounds checking
        for point in [start, end]:
            if not (0 <= point[0] < self.rows and 0 <= point[1] < self.cols):
                raise ValueError("Start or end position is out of bounds")

        # Handle start == end
        if start == end:
            return [start]

        # Handle start or end being a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Priority Queue: (priority, current_cost, current_node)
        # Priority = current_cost + heuristic
        open_set = []
        heapq.heappush(open_set, (0, 0, start))
        
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}

        while open_set:
            _, current_cost, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Explore 4-directional neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols:
                    weight = self.grid[neighbor[0]][neighbor[1]]
                    if weight == 0:  # Wall
                        continue
                    
                    tentative_g_score = current_cost + weight
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        g_score[neighbor] = tentative_g_score
                        priority = tentative_g_score + self._get_manhattan_distance(neighbor, end)
                        came_from[neighbor] = current
                        heapq.heappush(open_set, (priority, tentative_g_score, neighbor))

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
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 0))

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
    # Path A: (0,0) -> (0,1) -> (0,2) -> (1,2) | Total cost: 1+1+1 = 3
    # Path B: (0,0) -> (1,0) -> (1,1) -> (1,2) | Total cost: 1+10+1 = 12
    grid = [
        [1, 1, 1],
        [1, 10, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (1, 2))
    expected = [(0, 0), (0, 1), (0, 2), (1, 2)]
    assert path == expected

def test_start_or_end_is_wall():
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (1, 1)) is None

def test_complex_path():
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 1]
    ]
    astar = AStarGrid(grid)
    # Must go around the wall
    path = astar.find_path((0, 0), (3, 3))
    assert path is not None
    assert path[-1] == (3, 3)
    assert path[0] == (0, 0)