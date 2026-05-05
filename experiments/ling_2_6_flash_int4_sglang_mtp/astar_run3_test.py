from typing import Optional, List, Tuple
import heapq


class AStarGrid:
    """
    A class for performing A* pathfinding on a weighted 2D grid.

    Attributes:
        grid (List[List[int]]): A 2D list representing the grid. 
                                0 represents a wall, positive integers represent weights.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the AStarGrid with the given grid.

        Args:
            grid (List[List[int]]): A 2D list where 0 indicates a wall and 
                                    positive integers indicate traversal cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using the A* algorithm.

        Args:
            start (Tuple[int, int]): Starting coordinates (row, col).
            end (Tuple[int, int]): Target coordinates (row, col).

        Returns:
            Optional[List[Tuple[int, int]]]: A list of coordinates representing the path,
                                             or None if no path exists.

        Raises:
            ValueError: If start or end is out of bounds.
        """
        r1, c1 = start
        r2, c2 = end

        if not (0 <= r1 < self.rows and 0 <= c1 < self.cols):
            raise ValueError("Start position is out of bounds.")
        if not (0 <= r2 < self.rows and 0 <= c2 < self.cols):
            raise ValueError("End position is out of bounds.")

        if self.grid[r1][c1] == 0 or self.grid[r2][c2] == 0:
            return None  # Start or end is a wall

        if start == end:
            return [start]

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
            """Manhattan distance heuristic."""
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # Priority queue: (f_score, g_score, position)
        open_set = [(heuristic(start, end), 0, start)]
        came_from = {}

        # g_score: cost from start to current node
        g_score = {start: 0}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                nr, nc = neighbor

                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.grid[nr][nc] == 0:
                        continue  # Skip walls

                    tentative_g = current_g + self.grid[nr][nc]

                    if neighbor not in g_score or tentative_g < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        return None  # No path found

import pytest


def test_direct_path():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]


def test_wall_in_middle():
    grid = [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[-1] == (2, 2)


def test_start_is_end():
    grid = [
        [1, 2],
        [3, 4]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]


def test_out_of_bounds_raises_value_error():
    grid = [[1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))


def test_no_path_due_to_walls():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None


def test_weighted_path():
    grid = [
        [1, 100, 1],
        [1, 100, 1],
        [1,   1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]