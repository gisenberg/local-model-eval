import heapq
from typing import List, Tuple, Optional, Set

class AStarGrid:
    """A class to perform A* pathfinding on a 2D weighted grid."""

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the grid.
        :param grid: 2D list where 0 is impassable and >0 is movement cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _get_heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculates Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_valid(self, r: int, c: int) -> bool:
        """Checks if a cell is within bounds and not a wall."""
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] > 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using A*.
        :return: List of coordinates representing the path, or None if no path exists.
        :raises ValueError: If start or end coordinates are out of grid bounds.
        """
        # Bounds check
        for point in [start, end]:
            if not (0 <= point[0] < self.rows and 0 <= point[1] < self.cols):
                raise ValueError("Start or end coordinate out of bounds.")

        # Edge case: start or end is a wall
        if self.grid[start[0]][start[ors]] == 0 or self.grid[end[0]][end[1]] == 0:
            # Note: The requirement says return None if start or end is a wall
            if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
                return None

        # Edge case: start == end
        if start == end:
            return [start]

        # Priority Queue: (f_score, current_node)
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # Trackers
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: dict[Tuple[int, int], int] = {start: 0}
        
        # Directions: Up, Down, Left, Right
        neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]

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

            for dr, dc in neighbors:
                neighbor = (current[0] + dr, current[1] + dc)

                if self._is_valid(neighbor[0], neighbor[1]):
                    # Cost to enter the neighbor cell
                    tentative_g_score = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._get_heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score, neighbor))

        return None

# --- Pytest Suite ---
import pytest

def test_simple_path():
    """Simple path on a uniform grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path == [(0, 0), (0, 1), (0, 2)]

def test_path_around_obstacles():
    """Path must navigate around a wall (0)."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 0))
    # Must go around the center wall: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (2,1) -> (2,0)
    # Or similar. The shortest path in steps is actually (0,0)->(1,0) is blocked.
    # Let's use a simpler wall:
    grid2 = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar2 = AStarGrid(grid2)
    path = astar2.find_path((0, 0), (0, 2))
    assert (1, 0) not in path # Path must go down to row 2 to bypass the wall
    assert (0, 0) in path and (0, 2) in path

def test_weighted_grid_optimality():
    """Path prefers lower-cost cells even if longer distance."""
    grid = [
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Direct path (0,0) -> (1,0) -> (2,0) is cost 1+1=2. 
    # But if we go (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (2,1) -> (2,0)
    # The cost of entering (1,1) is 10.
    path = astar.find_path((0, 0), (2, 2))
    # Check that the path does not include the high-cost cell (1,1) if a cheaper detour exists
    assert (1, 1) not in path

def test_no_path_exists():
    """Path is impossible due to walls."""
    grid = [
        [1, 0, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) is None

def test_start_equals_end():
    """Start and end are the same coordinate."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_invalid_coordinates():
    """Coordinates outside the grid bounds should raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))

def test_start_is_wall():
    """If start or end is a wall, return None."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (1, 1)) is None