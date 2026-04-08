import heapq
from typing import List, Tuple, Optional, Dict


class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the grid.
        :param grid: 2D list where 0 is a wall and >0 is movement cost.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Returns valid 4-directional neighbors (up, down, left, right).
        """
        r, c = pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] > 0:
                    neighbors.append((nr, nc))
        return neighbors

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculates Manhattan distance between two points.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using A* algorithm.
        :return: List of (row, col) coordinates or None if no path exists.
        :raises ValueError: If start or end are out of bounds.
        """
        # Bounds checking
        for pos in [start, end]:
            if not (0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols):
                raise ValueError(f"Coordinate {pos} is out of bounds.")

        # Edge case: Start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Edge case: Start is end
        if start == end:
            return [start]

        # Priority Queue: (f_score, current_pos)
        open_set = []
        heapq.heappush(open_set, (0, start))

        # Tracking costs and paths
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            for neighbor in self._get_neighbors(current):
                # Cost to enter neighbor
                tentative_g_score = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))

        return None


# --- Pytest Tests ---
import pytest

def test_simple_uniform_grid():
    """Test a simple path on a grid with uniform costs."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Total cost calculation: sum of costs of cells in path excluding start (standard convention)
    # or including start? The prompt says "cost to enter that cell".
    # Path: (0,0)->(0,1)->(0,2)->(1,2)->(2,2). Costs: 1+1+1+1 = 4.
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4

def test_path_around_obstacles():
    """Test pathfinding that must navigate around walls."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 0), (1, 2))
    assert path is not None
    # Must go around the center wall (1,1)
    assert (1, 1) not in path
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4 # (1,0)->(0,0)->(0,1)->(0,2)->(1,2) is 4 steps

def test_weighted_grid_optimality():
    """Test that the path prefers lower-cost cells even if the route is longer."""
    grid = [
        [1, 1, 1, 1, 1],
        [1, 9, 9, 9, 1],
        [1, 1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Direct path through 9s would be (1,0)->(1,1)->(1,2)->(1,3)->(1,4) cost 36
    # Path around the edge: (1,0)->(0,0)->(0,1)->(0,2)->(0,3)->(0,4)->(1,4) cost 6
    path = astar.find_path((1, 0), (1, 4))
    assert path is not None
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 6 

def test_no_path_exists():
    """Test that None is returned when the destination is blocked."""
    grid = [
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) is None

def test_start_equals_end():
    """Test that [start] is returned if start is end."""
    grid = [[5]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_invalid_coordinates():
    """Test that ValueError is raised for out of bounds coordinates."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))