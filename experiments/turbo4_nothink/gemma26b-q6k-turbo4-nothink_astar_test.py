import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    Implements A* pathfinding on a 2D grid where cell values represent 
    the cost to enter that cell. 0 represents an impassable wall.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the grid.
        :param grid: A 2D list of integers representing movement costs.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _get_heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculates Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_valid(self, r: int, c: int) -> bool:
        """Checks if a coordinate is within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using A*.
        :param start: (row, col) starting position.
        :param end: (row, col) target position.
        :return: List of (row, col) tuples representing the path, or None if no path exists.
        :raises ValueError: If start or end are out of bounds.
        """
        if not self._is_valid(*start) or not self._is_valid(*end):
            raise ValueError("Start or end coordinates are out of bounds.")

        # If start or end is a wall, no path is possible
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        if start == end:
            return [start]

        # open_set stores (f_score, current_pos)
        open_set = []
        heapq.heappush(open_set, (0, start))

        # g_score[node] is the cost of the cheapest path from start to node currently known
        g_score = {start: 0}
        
        # came_from[node] stores the predecessor of node to reconstruct the path
        came_from = {}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            r, c = current
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (r + dr, c + dc)

                if self._is_valid(*neighbor):
                    cost = self.grid[neighbor[0]][neighbor[1]]
                    
                    # If cost is 0, it's a wall
                    if cost == 0:
                        continue

                    tentative_g_score = g_score[current] + cost

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._get_heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score, neighbor))

        return None

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstructs the path from the came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

# --- Pytest Tests ---

import pytest

def test_simple_path():
    """Test a simple path on a uniform grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Total cost: start is not counted in movement cost logic usually, 
    # but here we add cost of entering each cell. 
    # Path: (0,0)->(0,1)->(0,2)->(1,2)->(2,2). Costs: 1+1+1+1 = 4.
    # Wait, the prompt says "cost to enter that cell". 
    # Path length for (0,0) to (2,2) is 5 nodes. 
    # Entering (0,1), (0,2), (1,2), (2,2) = 4 steps * 1 cost = 4.
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4

def test_path_around_obstacles():
    """Test pathfinding around a wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 0), (1, 2))
    assert path is not None
    # Must go around the 0 at (1,1)
    assert (1, 1) not in path
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4 # (1,0)->(0,0)->(0,1)->(0,2)->(1,2) or similar

def test_weighted_grid():
    """Test that path prefers lower-cost cells."""
    # Path A: (0,0)->(0,1)->(0,2) cost: 1 + 10 = 11
    # Path B: (0,0)->(1,0)->(1,1)->(1,2)->(0,2) cost: 1 + 1 + 1 + 1 = 4
    grid = [
        [1, 10, 1],
        [1,  1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path == [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4

def test_no_path_exists():
    """Test when the target is fully blocked."""
    grid = [
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_start_equals_end():
    """Test when start and end are the same."""
    grid = [[5]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_invalid_coordinates():
    """Test that out of bounds raises ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))