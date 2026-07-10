import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """A* pathfinding algorithm on a weighted 2D grid."""

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid.

        :param grid: 2D list where 0 represents a wall and positive integers
                     represent traversal costs to enter that cell.
        :raises ValueError: If grid is empty.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within bounds and not a wall."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic for 4-directional movement."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        :param start: Starting coordinates (row, col)
        :param end: Ending coordinates (row, col)
        :return: List of coordinates representing the optimal path, or None if unreachable.
        :raises ValueError: If start/end are out of bounds or on walls.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds or on a wall.")

        if start == end:
            return [start]

        # Priority queue: (f_score, counter, g_score, position)
        # counter breaks ties to ensure FIFO behavior for equal f-scores
        counter = 0
        open_set = [(self._heuristic(start, end), counter, 0, start)]
        g_score = {start: 0}
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            f, _, g, current = heapq.heappop(open_set)

            # Skip stale entries (already found a better path to this node)
            if g > g_score[current]:
                continue

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
                if not self._is_valid(neighbor):
                    continue

                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, tentative_g, neighbor))

        return None

import pytest

def test_basic_pathfinding():
    """Test standard pathfinding on an open grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Optimal Manhattan distance + 1

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
        astar.find_path((0, 0), (2, 2))

def test_wall_positions_raise_value_error():
    """Test that start/end on walls raise ValueError."""
    grid = [[1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="on a wall"):
        astar.find_path((0, 1), (1, 1))  # Start on wall
    with pytest.raises(ValueError, match="on a wall"):
        astar.find_path((0, 0), (0, 1))  # End on wall

def test_weighted_optimal_path():
    """Test that A* chooses lower-cost path over shorter Manhattan distance."""
    # Middle column has high cost (10), forcing path around it
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))

    assert path is not None
    # Verify path avoids expensive cells
    assert (0, 1) not in path
    assert (1, 1) not in path

    # Calculate total traversal cost (excluding start cell)
    total_cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert total_cost == 4  # Optimal cost: 1+1+1+1