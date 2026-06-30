import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid cells with value 0 represent walls (impassable).
    Positive integer values represent the traversal cost (weight) of the cell.
    Movement is restricted to 4 directions (up, down, left, right).
    """
    def __init__(self, grid: List[List[int]]) -> None:
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_within_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within grid boundaries."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic for 4-directional movement."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using the A* algorithm.

        Args:
            start: (row, col) tuple of the starting position.
            end: (row, col) tuple of the target position.

        Returns:
            A list of (row, col) tuples representing the optimal path from start to end,
            or None if no valid path exists.

        Raises:
            ValueError: If start or end is out of bounds or lies on a wall (0).
        """
        if not self._is_within_bounds(start):
            raise ValueError("Start position is out of bounds.")
        if not self._is_within_bounds(end):
            raise ValueError("End position is out of bounds.")
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall.")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall.")

        if start == end:
            return [start]

        # Priority queue stores (f_score, counter, position)
        # Counter ensures stable sorting when f_scores are equal
        counter = 0
        open_set = [(0, counter, start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._is_within_bounds(neighbor):
                    continue
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue

                # Cost to enter the neighbor cell
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return None

import pytest

def test_basic_pathfinding():
    """Test simple path on an open grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Optimal length for 3x3 grid

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_exists():
    """Test when target is completely blocked by walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_start_end_raises_value_error():
    """Test that starting or ending on a wall raises ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((1, 1), (0, 0))

def test_weighted_optimal_path():
    """Test that A* chooses the lowest-cost path over the shortest geometric path."""
    # Center cell has high weight (9), forcing a detour
    grid = [
        [1, 1, 1],
        [1, 9, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert (1, 1) not in path  # Must avoid the heavy cell
    
    # Calculate total traversal cost (sum of weights of entered cells)
    cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert cost == 4  # Optimal detour cost: 1+1+1+1