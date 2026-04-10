import heapq
from typing import List, Tuple, Optional, Dict


class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the AStarGrid with a 2D list of movement costs.
        
        :param grid: 2D list where 0 represents an impassable wall and 
                     positive integers represent the cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculates the Manhattan distance between two points.
        
        :param a: The current cell coordinates (row, col).
        :param b: The target cell coordinates (row, col).
        :return: The Manhattan distance.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using the A* algorithm.

        :param start: Starting (row, col) coordinates.
        :param end: Ending (row, col) coordinates.
        :return: A list of (row, col) tuples representing the path, or None if no path exists.
        :raises ValueError: If start or end coordinates are out of grid bounds.
        """
        # 1. Bounds Check
        for point in [start, end]:
            if not (0 <= point[0] < self.rows and 0 <= point[1] < self.cols):
                raise ValueError(f"Coordinate {point} is out of bounds.")

        # 2. Wall Check
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # 3. Start equals End
        if start == end:
            return [start]

        # A* Algorithm Initialization
        # open_set stores (f_score, current_node)
        open_set: List[Tuple[int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score[n] is the cost of the cheapest path from start to n currently known
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        
        # f_score[n] = g_score[n] + heuristic(n, end)
        f_score: Dict[Tuple[int, int], float] = {start: self._heuristic(start, end)}

        while open_set:
            # Pop the node with the lowest f_score
            _, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            # Explore 4-directional neighbors
            r, c = current
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (r + dr, c + dc)
                nr, nc = neighbor

                # Check bounds and if neighbor is a wall
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] > 0:
                    # Cost to enter the neighbor
                    tentative_g_score = g_score[current] + self.grid[nr][nc]

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        # This path to neighbor is better than any previous one
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstructs the path from the came_from dictionary.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


# --- Pytest Tests ---

import pytest

def get_path_cost(grid, path):
    """Helper to calculate total cost of a path (excluding start cell)."""
    if not path or len(path) <= 1:
        return 0
    return sum(grid[r][c] for r, c in path[1:])

def test_simple_path():
    """Test a simple path on a uniform grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path == [(0, 0), (0, 1), (0, 2)]
    assert get_path_cost(grid, path) == 2

def test_path_around_obstacles():
    """Test finding a path around walls."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 1), (2, 1))
    # Path must go around the wall at (1, 1)
    assert (1, 1) not in path
    assert path is not None
    assert get_path_cost(grid, path) == 4 # (0,1)->(0,0)->(1,0)->(2,0)->(2,1) or similar

def test_weighted_grid_optimality():
    """Test that the path prefers lower-cost cells over shorter distance."""
    grid = [
        [1,  1,  1],
        [1, 10,  1],
        [1,  1,  1]
    ]
    astar = AStarGrid(grid)
    # Direct path (0,0)->(1,1)->(2,2) is impossible (4-dir), 
    # but (0,0)->(0,1)->(1,1)->(2,1)->(2,2) is very expensive.
    # Path should go around the edges.
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Cost of going through (1,1) would be much higher
    assert get_path_cost(grid, path) == 4 

def test_no_path_exists():
    """Test when the destination is completely blocked."""
    grid = [
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_start_equals_end():
    """Test when start and end are the same coordinate."""
    grid = [[5, 5], [5, 5]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_invalid_coordinates():
    """Test that out of bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_start_is_wall():
    """Test that starting on a wall returns None."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (1, 1)) is None