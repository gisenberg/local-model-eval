import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """A class to represent a 2D weighted grid for A* pathfinding."""

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the grid.
        
        :param grid: A 2D list where 0 represents a wall and positive integers represent movement cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

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

        # 3. Equality Check
        if start == end:
            return [start]

        # A* Initialization
        # open_set stores (f_score, (row, col))
        open_set: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, start))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score[node] is the cost of the cheapest path from start to node found so far
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        
        # f_score[node] = g_score[node] + heuristic(node, end)
        f_score: Dict[Tuple[int, int], float] = {start: float(self._heuristic(start, end))}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            # Explore 4-directional neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)

                if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols:
                    weight = self.grid[neighbor[0]][neighbor[1]]
                    
                    # If weight is 0, it's a wall
                    if weight == 0:
                        continue

                    tentative_g_score = g_score[current] + weight

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] quite = tentative_g_score
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstructs the path from the came_from dictionary.
        
        :param came_from: Dictionary mapping nodes to their predecessors.
        :param current: The end node of the path.
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
    """Test a simple path on a uniform grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Total cost: sum of weights of all cells except start
    cost = sum(grid[r][c] for r, c in path[1:])
    assert cost == 4  # (0,1), (0,2), (1,2), (2,2) -> 1+1+1+1

def test_path_around_obstacles():
    """Test finding a path around walls."""
    grid = [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 0))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]
    cost = sum(grid[r][c] for r, c in path[1:])
    assert cost == 6

def test_weighted_grid_optimality():
    """Test that the path prefers lower-cost cells even if longer."""
    grid = [
        [1,  1,  1],
        [1, 10,  1],
        [1,  1,  1]
    ]
    astar = AStarGrid(grid)
    # Path should go around the 10
    path = astar.find_path((0, 0), (2, 2))
    cost = sum(grid[r][c] for r, c in path[1:])
    # Optimal path: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) cost 4
    # Direct path through 10: (0,0)->(1,1)->(2,2) cost 11
    assert cost == 4
    assert (1, 1) not in path

def test_no_path_exists():
    """Test when the destination is fully blocked."""
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
        astar.find_path((0, 0), (5, 5))

def test_start_or_end_is_wall():
    """Test that starting or ending on a wall returns None."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (1, 1)) is None
    assert astar.find_path((1, 1), (0, 0)) is None