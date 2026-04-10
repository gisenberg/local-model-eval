from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the A* grid with a 2D list of movement costs.
        0 = impassable wall, positive int = cost to enter that cell.
        """
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        Returns the path as a list of coordinates or None if no path exists.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end is out of bounds.")
        if start == end:
            return [start]

        # Directions: up, down, left, right
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        # Priority queue: (f, g, x, y)
        open_set = [(0, 0, start[0], start[1])]
        heapq.heapify(open_set)

        # Keep track of visited cells and their g cost
        came_from = {}
        g_cost = {start: 0}
        f_cost = {start: self._heuristic(start, end)}

        while open_set:
            _, _, x, y = heapq.heappop(open_set)

            if (x, y) == end:
                # Reconstruct path
                path = [(x, y)]
                while (x, y) != start:
                    x, y = came_from[(x, y)]
                    path.append((x, y))
                return path[::-1]

            # Skip if we've already found a better path
            if (x, y) in g_cost and g_cost[(x, y)] < f_cost[(x, y)]:
                continue

            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if self._is_valid((nx, ny)) and self.grid[nx][ny] != 0:
                    new_g = g_cost[(x, y)] + self.grid[nx][ny]
                    if (nx, ny) not in g_cost or new_g < g_cost.get((nx, ny), float('inf')):
                        g_cost[(nx, ny)] = new_g
                        f = new_g + self._heuristic((nx, ny), end)
                        f_cost[(nx, ny)] = f
                        heapq.heappush(open_set, (f, new_g, nx, ny))
                        came_from[(nx, ny)] = (x, y)

        # No path found
        return None

    def _heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within bounds and not a wall."""
        x, y = pos
        return 0 <= x < self.height and 0 <= y < self.width and self.grid[x][y] != 0

import pytest


def test_find_path_simple():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]

def test_find_path_with_wall():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]

def test_find_path_start_end_same():
    grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_find_path_out_of_bounds():
    grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((3, 0), (2, 2))

def test_find_path_no_path():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_find_path_with_high_cost():
    grid = [
        [1, 10, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]