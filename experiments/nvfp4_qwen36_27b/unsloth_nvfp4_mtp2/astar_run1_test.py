import heapq
from typing import List, Tuple, Optional, Dict, Set

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.

    Uses 4-directional movement, Manhattan heuristic, and a priority queue (heapq).
    Grid cells with value 0 are treated as walls. Positive integers represent traversal costs.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid.

        Args:
            grid: 2D list where 0 represents a wall and positive integers
                  represent the traversal cost of that cell.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if coordinates are within grid boundaries."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate admissible Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start: Starting coordinates (row, col).
            end: Ending coordinates (row, col).

        Returns:
            List of coordinates from start to end, or None if no path exists.

        Raises:
            ValueError: If start or end is out of bounds or located on a wall.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds.")
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            raise ValueError("Start or end position is a wall.")

        if start == end:
            return [start]

        # Priority queue stores (f_score, tie_breaker, position)
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))

        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set: Set[Tuple[int, int]] = set()
        counter = 1

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                return path[::-1]

            if current in closed_set:
                continue
            closed_set.add(current)

            r, c = current
            # 4-directional movement: up, down, left, right
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)

                if not self._is_valid(neighbor) or self.grid[nr][nc] == 0:
                    continue

                move_cost = self.grid[nr][nc]
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
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Manhattan distance + 1 node

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_at_start_or_end_raises_value_error():
    """Test that walls at start/end positions raise ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))

    grid2 = [[1, 1], [1, 0]]
    astar2 = AStarGrid(grid2)
    with pytest.raises(ValueError, match="wall"):
        astar2.find_path((0, 0), (1, 1))

def test_no_path_returns_none():
    """Test that unreachable endpoints return None."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_weighted_optimality():
    """Test that A* chooses the lowest-cost path over the shortest geometric path."""
    # Direct path through middle has high weights (10),
    # but a longer path along the edge has lower total cost.
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Optimal path should avoid weight-10 cells
    assert (0, 1) not in path
    assert (1, 1) not in path
    assert path[-1] == (2, 2)