import heapq
from typing import List, Tuple, Optional, Dict


class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the grid.
        
        :param grid: A 2D list where 0 represents a wall and positive integers 
                     represent the cost to enter that cell.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _heuristic(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """
        Calculates the Manhattan distance between two points.
        """
        return abs(r1 - r2) + abs(c1 - c2)

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using the A* algorithm.

        :param start: (row, col) starting coordinates.
        :param end: (row, col) target coordinates.
        :return: List of (row, col) tuples representing the path, or None if no path exists.
        :raises ValueError: If start or end coordinates are out of grid bounds.
        """
        start_r, start_c = start
        end_r, end_c = end

        # 1. Bounds Check
        for r, c in [start, end]:
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(f"Coordinates ({r}, {c}) are out of bounds.")

        # 2. Wall Check
        if self.grid[start_r][start_c] == 0 or self.grid[end_r][end_c] == 0:
            return None

        # 3. Start equals End
        if start == end:
            return [start]

        # A* Algorithm Initialization
        # open_set stores (f_score, (row, col))
        open_set: List[Tuple[int, Tuple[int, int]]] = [(self._heuristic(start_r, start_c, end_r, end_c), start)]
        
        # came_from maps a node to its predecessor to reconstruct the path
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score[node] is the cost of the cheapest path from start to node currently known
        g_score: Dict[Tuple[int, int], int] = {start: 0}

        while open_set:
            # Pop the node with the lowest f_score
            current_f, current = heapq.heappop(open_set)
            curr_r, curr_c = current

            # If we reached the end, reconstruct the path
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Explore 4-directional neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = curr_r + dr, curr_c + dc
                neighbor = (nr, nc)

                # Check bounds and if the cell is a wall
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] > 0:
                    # Cost to enter the neighbor cell
                    tentative_g_score = g_score[current] + self.grid[nr][nc]

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        # This path to neighbor is better than any previous one
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._heuristic(nr, nc, end_r, end_c)
                        heapq.heappush(open_set, (f_score, neighbor))

        return None


# --- Pytest Tests ---
import pytest

def calculate_path_cost(grid, path):
    """Helper to calculate the total cost of a path (excluding start cell cost)."""
    if not path or len(path) <= 1:
        return 0
    return sum(grid[r][c] for r, c in path[1:])

def test_simple_path():
    """Test a simple path on a uniform grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert calculate_path_cost(grid, path) == 4  # (0,1), (0,2), (1,2), (2,2) or similar

def test_path_around_obstacles():
    """Test pathfinding around a wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert (1, 1) not in path
    assert calculate_path_cost(grid, path) == 4

def test_weighted_grid():
    """Test that the path prefers lower-cost cells over shorter paths."""
    grid = [
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Path through (1,1) would cost 10 + 1 = 11. 
    # Path around the edge costs 1+1+1+1 = 4.
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert (1, 1) not in path
    assert calculate_path_cost(grid, path) == 4

def test_no_path_exists():
    """Test when the destination is completely blocked."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_start_equals_end():
    """Test when start and end are the same coordinate."""
    grid = [[5]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_invalid_coordinates():
    """Test that out of bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_start_or_end_is_wall():
    """Test that starting or ending on a wall returns None."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (1, 1)) is None
    assert astar.find_path((1, 1), (0, 0)) is None