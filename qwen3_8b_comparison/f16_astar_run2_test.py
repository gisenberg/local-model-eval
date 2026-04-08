from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.

    Attributes:
        grid (List[List[int]]): The 2D grid where each cell's value represents the movement cost.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid.

        Args:
            grid (List[List[int]]): The 2D grid where each cell's value represents the movement cost.
        """
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.

        Args:
            start (Tuple[int, int]): Starting position (row, col)
            end (Tuple[int, int]): Target position (row, col)

        Returns:
            Optional[List[Tuple[int, int]]]: The shortest path as a list of coordinates, or None if no path exists.
        """
        # Validate input
        if not self._is_valid_position(start) or not self._is_valid_position(end):
            raise ValueError("Start or end position is out of bounds.")
        if start == end:
            return [start]

        # Initialize open set and closed set
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, end)}

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

            # Explore neighbors
            for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + direction[0], current[1] + direction[1])
                if not self._is_valid_position(neighbor):
                    continue
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue  # Wall

                tentative_g_score = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        return None

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Manhattan distance heuristic between two points.

        Args:
            a (Tuple[int, int]): First point (row, col)
            b (Tuple[int, int]): Second point (row, col)

        Returns:
            int: Manhattan distance between a and b
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_valid_position(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is within the grid bounds and not a wall.

        Args:
            position (Tuple[int, int]): Position to check

        Returns:
            bool: True if position is valid, False otherwise
        """
        row, col = position
        return 0 <= row < self.height and 0 <= col < self.width and self.grid[row][col] != 0

import pytest
from typing import List, Tuple, Optional


def test_simple_path():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]

def test_start_end_same():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_no_path():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_wall():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_out_of_bounds():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 3), (0, 0))

def test_invalid_start_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((3, 0), (0, 0))

if __name__ == "__main__":
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    print("Path found:", path)