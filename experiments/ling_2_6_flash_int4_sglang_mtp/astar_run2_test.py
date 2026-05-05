from typing import Optional, List, Tuple
import heapq


class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.

    Attributes:
        grid (List[List[int]]): A 2D list representing the grid. 
                                0 indicates a wall, positive integers indicate weights.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D grid.

        Args:
            grid (List[List[int]]): A 2D list of integers where 0 represents a wall,
                                    and positive integers represent traversal cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start (Tuple[int, int]): Starting coordinate (row, col).
            end (Tuple[int, int]): Target coordinate (row, col).

        Returns:
            Optional[List[Tuple[int, int]]]: The optimal path as a list of coordinates,
                                             or None if no path exists.

        Raises:
            ValueError: If start or end is out of bounds or on a wall.
        """
        r1, c1 = start
        r2, c2 = end

        # Validate bounds
        if not (0 <= r1 < self.rows and 0 <= c1 < self.cols):
            raise ValueError("Start position is out of bounds.")
        if not (0 <= r2 < self.rows and 0 <= c2 < self.cols):
            raise ValueError("End position is out of bounds.")

        # Check if start or end is a wall
        if self.grid[r1][c1] == 0 or self.grid[r2][c2] == 0:
            raise ValueError("Start or end is on a wall.")

        # Early exit if start == end
        if start == end:
            return [start]

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Priority queue: (f_score, g_score, position)
        open_set = [(self._heuristic(start, end), 0, start)]
        came_from = {}

        # g_score: cost from start to current node
        g_score = {start: 0}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)

                # Skip out-of-bounds
                if not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols):
                    continue

                # Skip walls
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue

                # Calculate tentative g_score
                tentative_g = current_g + self.grid[neighbor[0]][neighbor[1]]

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        # No path found
        return None

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Manhattan distance heuristic.

        Args:
            a (Tuple[int, int]): First coordinate.
            b (Tuple[int, int]): Second coordinate.

        Returns:
            int: Manhattan distance between a and b.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from came_from map.

        Args:
            came_from (dict): Mapping from node to its predecessor.
            current (Tuple[int, int]): Current node (end node).

        Returns:
            List[Tuple[int, int]]: Reconstructed path from start to end.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

import pytest


def test_direct_path():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]


def test_path_with_wall():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]


def test_start_equals_end():
    grid = [[5]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]


def test_out_of_bounds_raises_error():
    grid = [[1, 2], [3, 4]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))


def test_start_on_wall_raises_error():
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))


def test_no_path_exists():
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) is None