import heapq
from typing import List, Tuple, Dict, Optional, Set

class AStarGrid:
    """
    A class representing a 2D grid for A* pathfinding.
    
    Attributes:
        grid (List[List[int]]): A 2D list where 0 is impassable and >0 is movement cost.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the AStarGrid with a weighted 2D grid.

        Args:
            grid (List[List[int]]): 2D list of movement costs.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculates the Manhattan distance between two points.

        Args:
            a (Tuple[int, int]): First coordinate (row, col).
            b (Tuple[int, int]): Second coordinate (row, col).

        Returns:
            int: Manhattan distance.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from start to end using the A* algorithm.

        Args:
            start (Tuple[int, int]): Starting coordinates (row, col).
            end (Tuple[int, int]): Destination coordinates (row, col).

        Returns:
            Optional[List[Tuple[int, int]]]: List of coordinates from start to end, 
                                              or None if no path exists.

        Raises:
            ValueError: If start or end coordinates are out of grid bounds.
        """
        start_r, start_c = start
        end_r, end_c = end

        # 1. Bounds Check
        for r, c in [start, end]:
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError("Start or end coordinate is out of bounds.")

        # 2. Wall Check (0 is impassable)
        if self.grid[start_r][start_c] == 0 or self.grid[end_r][end_c] == 0:
            return None

        # 3. Start equals End
        if start == end:
            return [start]

        # A* Algorithm Initialization
        # open_set stores (f_score, current_node)
        open_set: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score[node] is the cost of the cheapest path from start to node
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        
        # f_score[node] = g_score[node] + heuristic(node, end)
        f_score: Dict[Tuple[int, int], float] = {start: float(self._heuristic(start, end))}

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

            curr_r, curr_c = current

            # 4-directional movement: Up, Down, Left, Right
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (curr_r + dr, curr_c + dc)
                nr, nc = neighbor

                # Check bounds and if neighbor is a wall
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    cost = self.grid[nr][nc]
                    if cost == 0:
                        continue

                    # Tentative g_score is the cost to reach neighbor via current
                    tentative_g = g_score[current] + cost

                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor]                        = tentative_g
                        f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

# --- Pytest Suite ---
import pytest

def test_simple_path():
    """Test a simple path on a uniform grid."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (1, 1))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (1, 1)
    # Cost: entering (0,1) then (1,1) -> 1 + 1 = 2
    cost = sum(grid[r][c] for r, c in path[1:])
    assert cost == 2

def test_path_around_obstacles():
    """Test path finding around a wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    # Must go around the 0 at (1,1)
    assert (1, 1) not in path
    assert len(path) > 3 # Path must be longer to avoid center

def test_weighted_grid_preference():
    """Test that path prefers lower-cost cells even if longer."""
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    # Direct path (0,0)->(0,1)->(0,2) cost is 10+1=11
    # Detour (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2) cost is 1+1+1+1+1+1=6
    cost = sum(grid[r][c] for r, c in path[1:])
    assert cost == 6
    assert (0, 1) not in path

def test_no_path_exists():
    """Test scenario where end is fully blocked."""
    grid = [
        [1, 0, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_start_equals_end():
    """Test scenario where start and end are the same."""
    grid = [[5]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_invalid_coordinates():
    """Test out of bounds and wall start/end."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    # Out of bounds
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))
        
    # Start or end is a wall
    grid_with_wall = [[1, 0], [1, 1]]
    astar_wall = AStarGrid(grid_with_wall)
    assert astar_wall.find_path((0, 1), (1, 1)) is None
    assert astar_wall.find_path((0, 0), (0, 1)) is None