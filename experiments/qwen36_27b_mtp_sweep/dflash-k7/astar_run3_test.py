import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """A* pathfinding algorithm on a weighted 2D grid."""

    def __init__(self, grid: List[List[float]]):
        """
        Initialize the grid.

        Args:
            grid: 2D list where 0 represents a wall and positive numbers 
                  represent the traversal cost to enter that cell.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.

        Args:
            start: (row, col) tuple for starting position.
            end: (row, col) tuple for ending position.

        Returns:
            List of (row, col) tuples representing the optimal path, 
            or None if no valid path exists.

        Raises:
            ValueError: If start/end are out of bounds or located on walls.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds.")

        # Validate walls
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall.")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall.")

        if start == end:
            return [start]

        # A* initialization
        open_set = []
        start_f = self._heuristic(start, end)
        heapq.heappush(open_set, (start_f, 0, start))
        
        g_score = {start: 0}
        came_from = {}
        counter = 1  # Tie-breaker for heapq to avoid tuple comparison issues

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 4-directional

        while open_set:
            current_f, _, current = heapq.heappop(open_set)

            # Lazy deletion: skip if we've already found a better path to this node
            if current_f > g_score[current] + self._heuristic(current, end):
                continue

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            r, c = current
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != 0:
                    tentative_g = g_score[current] + self.grid[nr][nc]
                    if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                        g_score[(nr, nc)] = tentative_g
                        came_from[(nr, nc)] = current
                        f_score = tentative_g + self._heuristic((nr, nc), end)
                        heapq.heappush(open_set, (f_score, counter, (nr, nc)))
                        counter += 1

        return None

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic for 4-directional movement."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ========================
# Pytest Test Suite
# ========================
import pytest

def test_basic_path():
    """Test standard pathfinding on an open grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Verify 4-directional adjacency
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[5]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_no_path():
    """Test when end is unreachable due to walls."""
    grid = [[1, 0, 1], [1, 0, 1], [1, 0, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None

def test_out_of_bounds_start():
    """Test ValueError when start is out of grid bounds."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="Start position is out of bounds"):
        astar.find_path((-1, 0), (1, 1))

def test_wall_at_end():
    """Test ValueError when end position is a wall."""
    grid = [[1, 0], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="End position is a wall"):
        astar.find_path((0, 0), (0, 1))

def test_weighted_optimal():
    """Test that A* chooses the lowest-cost path over the shortest-step path."""
    # Direct path through center has weight 10, detour has weight 1
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Verify path avoids high-weight cells
    assert all(grid[r][c] != 10 for r, c in path)
    # Verify total traversal cost is optimal (4 steps of weight 1)
    total_cost = sum(grid[r][c] for r, c in path[1:])  # exclude start node
    assert total_cost == 4