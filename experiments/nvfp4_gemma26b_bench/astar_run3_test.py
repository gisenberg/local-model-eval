import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A class representing a 2D grid for A* pathfinding.
    The grid contains weights where 0 represents a wall and 1 or higher represents traversable cost.
    """

    def __init__(self, grid: List[List[float]]):
        """
        Initialize the grid.
        :param grid: A 2D list of numbers. 0 is a wall, > 0 is the cost to enter that cell.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if a coordinate is within bounds and not a wall."""
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] > 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using A* algorithm.
        
        :param start: (row, col) starting position.
        :param end: (row, col) target position.
        :return: List of (row, col) tuples representing the path, or None if no path exists.
        :raises ValueError: If start or end are out of bounds or on a wall.
        """
        # Validation
        for pos in [start, end]:
            if not (0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols):
                raise ValueError(f"Position {pos} is out of bounds.")
            if self.grid[pos[0]][pos[1]] == 0:
                raise ValueError(f"Position {pos} is a wall.")

        if start == end:
            return [start]

        # Priority Queue stores: (priority, current_cost, current_node)
        # priority = current_cost + heuristic
        open_set: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, 0.0, start))

        # Track the best cost to reach a node and the parent for reconstruction
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        while open_set:
            _, current_cost, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            # If we found a better way to this node already, skip
            if current_cost > g_score.get(current, float('inf')):
                continue

            r, c = current
            # 4-directional movement
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (r + dr, c + dc)

                if self._is_valid(neighbor[0], neighbor[1]):
                    # Cost to enter the neighbor cell
                    weight = self.grid[neighbor[0]][neighbor[1]]
                    tentative_g_score = current_cost + weight

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        # Manhattan distance heuristic
                        h = abs(neighbor[0] - end[0]) + abs(neighbor[1] - end[1])
                        f_score = tentative_g_score + h
                        heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

        return None

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        return path[::-1]

# --- Pytest Tests ---

import pytest

def test_path_found():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert (0, 0) in path
    assert (2, 2) in path

def test_start_is_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_no_path_due_to_walls():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Path from top-left to top-right is blocked by wall at (0,1) and (1,1)
    # But (2,1) is open. However, if we block (0,1) and (1,1), (0,2) is unreachable from (0,0)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_weighted_path_preference():
    # Path through (0,1) is cost 10, path through (1,0) is cost 1
    grid = [
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    # Should go (0,0) -> (1,0) -> (1,1) -> (1,2) -> (0,2) to avoid the 10 weight
    assert (1, 1) in path

def test_out_of_bounds_raises_error():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_start_on_wall_raises_error():
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))

if __name__ == "__main__":
    # Manual verification if not running via pytest
    grid = [[1, 1, 1], [1, 5, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    print(f"Path: {astar.find_path((0,0), (2,2))}")