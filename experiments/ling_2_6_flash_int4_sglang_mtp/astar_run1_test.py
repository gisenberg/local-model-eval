from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    """
    A class for performing A* pathfinding on a weighted 2D grid.

    Attributes:
        grid (List[List[int]]): A 2D list representing the grid. 
                                0 represents a wall (impassable), 
                                positive integers represent passable cell weights.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the AStarGrid with the given 2D grid.

        Args:
            grid (List[List[int]]): A 2D list of integers where 0 is a wall
                                    and any positive integer is a cell weight.
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
            ValueError: If start or end is out of bounds or is a wall.
        """
        # Validate coordinates
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End is out of bounds.")
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start is a wall.")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End is a wall.")

        # Early exit if start == end
        if start == end:
            return [start]

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Heuristic function (Manhattan distance)
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
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

                # Check bounds and wall
                if not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols):
                    continue
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue

                # Calculate tentative g score
                weight = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = current_g + weight

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        # No path found
        return None

import pytest

def test_direct_path():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (1, 1), (2, 2)] or len(path) == 5  # Manhattan path

def test_wall_block():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 0))
    assert path is None

def test_start_equals_end():
    grid = [[5]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_out_of_bounds():
    grid = [[1, 2], [3, 4]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))

def test_start_is_wall():
    grid = [
        [0, 2],
        [3, 4]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))

def test_weighted_path():
    grid = [
        [1, 2, 3],
        [4, 5, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    # Optimal path should avoid high-cost center
    assert path == [(0, 0), (0, 1), (0, 2)] or path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]