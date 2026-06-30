import heapq
from typing import List, Tuple, Optional, Sequence


class AStarGrid:
    """A* pathfinding algorithm on a weighted 2D grid."""

    def __init__(self, grid: Sequence[Sequence[float]]):
        """
        Initialize the grid for pathfinding.

        Args:
            grid: A 2D sequence where 0 represents a wall/impassable cell,
                  and positive numbers represent the traversal cost to enter that cell.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within bounds and not a wall."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic (admissible and consistent for grid graphs)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* search.

        Args:
            start: (row, col) tuple of the starting position.
            end: (row, col) tuple of the target position.

        Returns:
            A list of (row, col) tuples representing the optimal path, 
            or None if no valid path exists.

        Raises:
            ValueError: If start or end is out of bounds or positioned on a wall.
        """
        if not self._is_valid(start):
            raise ValueError("Start position is out of bounds or on a wall.")
        if not self._is_valid(end):
            raise ValueError("End position is out of bounds or on a wall.")

        if start == end:
            return [start]

        # Priority queue: (f_score, counter, position)
        # Counter ensures stable ordering and avoids tuple comparison errors
        open_set = []
        counter = 0
        heapq.heappush(open_set, (self._heuristic(start, end), counter, start))
        counter += 1

        g_score = {start: 0}
        came_from = {}
        closed_set = set()

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            f, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue

            if current == end:
                # Reconstruct path
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                return path[::-1]

            closed_set.add(current)

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._is_valid(neighbor):
                    continue

                # Cost to enter the neighbor cell
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1

        return None

import pytest


def test_basic_pathfinding():
    """Test standard pathfinding on an unweighted grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Verify 4-directional adjacency
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1


def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]


def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))


def test_wall_at_start_or_end_raises_value_error():
    """Test that start/end on walls raise ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))
        
    grid2 = [[1, 1], [1, 0]]
    astar2 = AStarGrid(grid2)
    with pytest.raises(ValueError):
        astar2.find_path((0, 0), (1, 1))


def test_no_path_returns_none():
    """Test that completely blocked paths return None."""
    grid = [[1, 0, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None


def test_weighted_optimal_path():
    """Test that A* chooses the lowest-cost path over the shortest-step path."""
    # Direct path: (0,0)->(0,1)->(0,2) costs 100 + 1 = 101
    # Detour path: (0,0)->(1,0)->(1,1)->(1,2)->(0,2) costs 1+1+1+1 = 4
    grid = [
        [1, 100, 1],
        [1, 1, 1],
        [1, 100, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is not None
    assert (0, 1) not in path  # Avoids heavy cell
    assert (1, 1) in path     # Takes optimal detour