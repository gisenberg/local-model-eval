import heapq
from typing import List, Tuple, Optional, Dict, Set

class AStarGrid:
    """
    A class to represent a 2D grid for A* pathfinding.
    Grid values represent the cost to enter a cell (0 = impassable).
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

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Returns valid 4-directional neighbors of a cell."""
        r, c = pos
        neighbors = []
        # Up, Down, Left, Right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] > 0:  # Only add if not a wall
                    neighbors.append((nr, nc))
        return neighbors

    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculates the Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using A* algorithm.

        :param start: (row, col) starting coordinates.
        :param end: (row, col) target coordinates.
        :return: List of (row, col) tuples representing the path, or None if no path exists.
        :raises ValueError: If start or end coordinates are out of bounds.
        """
        # Bounds check
        for point in [start, end]:
            if not (0 <= point[0] < self.rows and 0 <= point[1] < self.cols):
                raise ValueError("Start or end coordinate is out of bounds.")

        # Edge case: Start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Edge case: Start is end
        if start == end:
            return [start]

        # Priority Queue: (f_score, current_node)
        open_set = []
        heapq.heappush(open_set, (0, start))

        # Tracking dictionaries
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        
        # Set for faster lookup in open_set
        open_set_hash: Set[Tuple[int, int]] = {start}

        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for neighbor in self._get_neighbors(current):
                # Cost to enter the neighbor cell
                tentative_g_score = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self._manhattan_distance(neighbor, end)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score, neighbor))
                        open_set_hash.add(neighbor)

        return None

# --- Pytest Suite ---
import pytest

def test_simple_path():
    """Test a simple path on a uniform grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)] or \
           path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    # Verify cost: 1 (start) + 1 + 1 + 1 + 1 (end) = 5 (Wait, cost is to enter cell)
    # Path: (0,0) -> (0,1) [cost 1] -> (0,2) [cost 1] -> (1,2) [cost 1] -> (2,2) [cost 1]
    # Total cost = sum of grid[r][c] for all cells in path EXCEPT start.
    # Let's check the sum of costs of cells in path excluding start.
    cost = sum(grid[r][c] for r, c in path[1:])
    assert cost == 4

def test_path_around_obstacles():
    """Test pathfinding when a wall blocks the direct route."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 1), (2, 1))
    # Must go around the 0 at (1, 1)
    assert (1, 1) not in path
    assert len(path) == 5 # (0,1) -> (0,0) -> (1,0) -> (2,0) -> (2,1) or similar

def test_weighted_grid():
    """Test that the algorithm prefers lower-cost cells even if the path is longer."""
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    # Direct path (0,0)->(0,1)->(0,2) cost is 10+1=11
    # Long path (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2) cost is 1+1+1+1+1+1=6
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]

def test_no_path_exists():
    """Test when the destination is completely blocked."""
    grid = [
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) is None

def test_start_equals_end():
    """Test when start and end are the same coordinate."""
    grid = [[5]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_invalid_coordinates():
    """Test that out of bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))