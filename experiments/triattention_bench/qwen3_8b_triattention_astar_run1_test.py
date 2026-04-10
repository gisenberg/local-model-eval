from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid with a 2D list of movement costs.
        0 represents impassable walls, positive integers represent movement cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        Returns the path as a list of coordinates, or None if no path exists.
        Raises ValueError if start or end is out of bounds.
        """
        # Edge case: start or end is out of bounds
        if not self.is_valid(start) or not self.is_valid(end):
            raise ValueError("Start or end is out of bounds")
        
        # Edge case: start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        
        # Edge case: start == end
        if start == end:
            return [start]

        # Initialize A* algorithm
        open_set = []
        start_row, start_col = start
        start_cost = self.grid[start_row][start_col]
        h = abs(start_row - end[0]) + abs(start_col - end[1])  # Manhattan distance
        heapq.heappush(open_set, (start_cost + h, start_cost, start))
        g_scores = {start: start_cost}
        came_from = {}

        while open_set:
            f, g, current = heapq.heappop(open_set)
            current_row, current_col = current

            # Check if we reached the end
            if current == end:
                # Reconstruct the path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            # Explore neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = current_row + dr, current_col + dc

                # Check if new position is within bounds
                if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                    # Skip walls
                    if self.grid[new_row][new_col] == 0:
                        continue

                    # Calculate tentative g score
                    tentative_g = g + self.grid[new_row][new_col]
                    new_node = (new_row, new_col)

                    # Update if this path is better
                    if new_node not in g_scores or tentative_g < g_scores[new_node]:
                        g_scores[new_node] = tentative_g
                        h_new = abs(new_row - end[0]) + abs(new_col - end[1])
                        f_new = tentative_g + h_new
                        heapq.heappush(open_set, (f_new, tentative_g, new_node))
                        came_from[new_node] = current

        # No path found
        return None

    def is_valid(self, pos: Tuple[int, int]) -> bool:
        """
        Check if a position is within the grid bounds.
        """
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols

import pytest


def test_simple_path():
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

def test_path_around_obstacles():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

def test_weighted_grid():
    grid = [
        [1, 3, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]

def test_no_path():
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_start_equals_end():
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_invalid_coordinates():
    grid = [[1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((1, 0), (0, 0))