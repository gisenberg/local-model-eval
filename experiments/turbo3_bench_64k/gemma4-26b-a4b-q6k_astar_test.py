import heapq
from typing import List, Tuple, Optional, Dict


class AStarGrid:
    """
    A class to represent a weighted 2D grid and perform A* pathfinding.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the AStarGrid with a 2D list of movement costs.
        
        :param grid: A 2D list where 0 represents a wall and positive integers represent movement cost.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculates the Manhattan distance between two points.
        
        :param a: Starting coordinate (row, col).
        :param b: Ending coordinate (row, col).
        :return: Manhattan distance.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from start to end using the A* algorithm.
        
        :param start: Starting coordinate (row, col).
        :param end: Ending coordinate (row, col).
        :return: List of (row, col) coordinates representing the path, or None if no path exists.
        :raises ValueError: If start or end coordinates are out of bounds.
        """
        # 1. Bounds Check
        for r, c in [start, end]:
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError("Start or end coordinates are out of bounds.")

        # 2. Wall Check
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # 3. Start is End Check
        if start == end:
            return [start]

        # A* Initialization
        # open_set stores (f_score, (row, col))
        open_set: List[Tuple[int, Tuple[int, int]]] = [(self._heuristic(start, end), start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score: cost from start to current node
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        
        # f_score: estimated total cost (g_score + heuristic)
        f_score: Dict[Tuple[int, int], float] = {start: float(self._heuristic(start, end))}

        while open_set:
            # Get node with lowest f_score
            _, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

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

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        # This path to neighbor is better than any previous one
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                        
                        # Add to open set if not already there (or if we found a better path)
                        # Note: heapq doesn't support updating, so we just push the new score
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstructs the path from the came_from dictionary.
        
        :param came_from: Dictionary mapping nodes to their predecessors.
        :param current: The end node.
        :return: List of coordinates from start to end.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


# --- Pytest Tests ---
import pytest

def test_simple_path():
    """Test a simple path on a uniform grid with all costs = 1."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (1, 1))
    assert path is not None
    # Path should be length 3 (start, middle, end) and cost should be 2
    assert len(path) == 3
    cost = sum(grid[r][c] for r, c in path[1:])
    assert cost == 2

def test_path_around_obstacles():
    """Test finding a path around a wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Must go around the wall at (1, 1)
    assert (1, 1) not in path
    cost = sum(grid[r][c] for r, c in path[1:])
    assert cost == 4  # (0,0)->(0,1)->(0,2)->(1,2)->(2,2) or similar

def test_weighted_grid_optimality():
    """Test that the path prefers lower-cost cells even if the path is longer."""
    grid = [
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    # The direct path through (1,1) costs 10+1=11. 
    # The path around costs 1+1+1+1=4.
    assert path is not None
    assert (1, 1) not in path
    cost = sum(grid[r][c] for r, c in path[1:])
    assert cost == 4

def test_no_path_exists():
    """Test that None is returned when the destination is blocked."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_start_equals_end():
    """Test that if start is end, it returns [start]."""
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