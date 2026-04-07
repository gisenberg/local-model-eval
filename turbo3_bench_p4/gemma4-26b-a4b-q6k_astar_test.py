import heapq
from typing import List, Tuple, Optional, Dict


class AStarGrid:
    """
    A class to represent a weighted 2D grid and perform A* pathfinding.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the AStarGrid with a 2D list of movement costs.
        0 represents an impassable wall.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculates the Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from start to end using the A* algorithm.
        Returns a list of (row, col) coordinates or None if no path exists.
        """
        # 1. Bounds and Wall Validation
        for point in [start, end]:
            if not (0 <= point[0] < self.rows and 0 <= point[1] < self.cols):
                raise ValueError(f"Coordinate {point} is out of bounds.")
            if self.grid[point[0]][point[1]] == 0:
                return None

        # 2. Edge Case: Start is End
        if start == end:
            return [start]

        # 3. Initialization
        # open_set stores (priority, current_node)
        open_set: List[Tuple[int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score: cost from start to current node
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        
        # f_score: estimated total cost (g_score + heuristic)
        f_score: Dict[Tuple[int, int], float] = {start: self._manhattan_distance(start, end)}

        while open_set:
            # Pop node with lowest f_score
            _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # 4-directional movement (Up, Down, Left, Right)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dr, current[1] + dc)

                # Check bounds and if neighbor is a wall
                if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols:
                    weight = self.grid[neighbor[0]][neighbor[1]]
                    if weight == 0:
                        continue

                    # Calculate tentative g_score
                    tentative_g_score = g_score[current] + weight

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        # This path to neighbor is better than any previous one
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._manhattan_distance(neighbor, end)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None


# --- Pytest Tests ---
import pytest

def test_simple_path():
    """Test a simple path on a uniform grid."""
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Total cost: entering (0,1), (0,2), (1,2), (2,2) -> 1+1+1+1 = 4
    # Or any path of length 5 (4 steps)
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4

def test_path_around_obstacles():
    """Test finding a path around a wall."""
    grid = [[1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 0), (1, 3))
    assert path is not None
    # Must go around the wall at row 1
    for r, c in path:
        assert grid[r][c] != 0
    assert path[0] == (1, 0)
    assert path[-1] == (1, 3)

def test_weighted_grid():
    """Test that the path prefers lower-cost cells even if longer."""
    grid = [[1, 1, 1, 1],
            [1, 10, 10, 1],
            [1, 1, 1, 1]]
    astar = AStarGrid(grid)
    # Direct path through (1,1) is blocked by high cost
    path = astar.find_path((0, 0), (2, 0))
    # Should go (0,0) -> (0,1) -> (0,2) -> (0,3) -> (1,3) -> (2,3) -> (2,2) -> (2,1) -> (2,0) 
    # or similar to avoid the 10s.
    # Actually, the shortest path is (0,0)->(1,0)->(2,0) if we don't hit 10s.
    # Let's make a better test:
    grid2 = [[1, 1, 1],
              [1, 10, 1],
              [1, 1, 1]]
    astar2 = AStarGrid(grid2)
    path2 = astar2.find_path((0, 1), (2, 1))
    # Path should be (0,1)->(0,0)->(1,0)->(2,0)->(2,1) or (0,1)->(0,2)->(1,2)->(2,2)->(2,1)
    # Cost via middle: 10. Cost via side: 1+1+1+1 = 4.
    total_cost = sum(grid2[r][c] for r, c in path2[1:])
    assert total_cost == 4
    assert (1, 1) not in path2

def test_no_path_exists():
    """Test when the end is completely blocked."""
    grid = [[1, 1, 1],
            [1, 0, 0],
            [1, 0, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_start_equals_end():
    """Test when start and end are the same coordinate."""
    grid = [[1, 1],
            [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_invalid_coordinates():
    """Test that out of bounds raises ValueError."""
    grid = [[1, 1],
            [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))

def test_start_is_wall():
    """Test that starting on a wall returns None."""
    grid = [[0, 1],
            [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (1, 1)) is None