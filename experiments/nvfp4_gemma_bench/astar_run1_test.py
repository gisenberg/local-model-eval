import heapq
from typing import List, Tuple, Optional, Dict

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
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _get_manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using A*.
        :return: List of coordinates from start to end, or None if no path exists.
        :raises ValueError: If start or end are out of bounds.
        """
        # Bounds checking
        for point in [start, end]:
            if not (0 <= point[0] < self.rows and 0 <= point[1] < self.cols):
                raise ValueError(f"Point {point} is out of grid bounds.")

        if start == end:
            return [start]

        # Priority Queue stores: (f_score, current_node)
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # Maps child -> parent to reconstruct path
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        
        # g_score: cost from start to current node
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        
        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                path = []
                while current:
                    path.append(current)
                    current = came_from.get(current)
                return path[::-1]

            # 4-directional movement
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)

                if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols:
                    weight = self.grid[neighbor[0]][neighbor[1]]
                    
                    # Wall check
                    if weight == 0:
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
    # Path A: (0,0)->(0,1)->(0,2) cost = 1+1 = 2
    # Path B: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2) cost = 1+1+1+1+1+1 = 6
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    # Put a wall at (1,1) to force a choice
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path == [(0, 0), (0, 1), (0, 2)]

def test_complex_maze():
    grid = [
        [1, 1, 1, 1],
        [0, 0, 0, 1],
        [1, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (4, 0))
    expected = [(0,0), (0,1), (0,2), (0,3), (1,3), (2,3), (2,2), (2,1), (2,0), (3,0), (4,0)]
    assert path == expected