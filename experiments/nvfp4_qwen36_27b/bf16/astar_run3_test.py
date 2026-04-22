import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Walls are represented by 0, and traversable cells have positive integer weights.
    Movement is restricted to 4 directions (up, down, left, right).
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid for pathfinding.

        Args:
            grid: 2D list where 0 represents a wall and positive integers represent cell weights.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty and rectangular.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start: Tuple (row, col) of the starting position.
            end: Tuple (row, col) of the target position.

        Returns:
            List of tuples representing the optimal path from start to end, 
            or None if no valid path exists.

        Raises:
            ValueError: If start or end is out of bounds or positioned on a wall.
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

        # Handle trivial case
        if start == end:
            return [start]

        # 4-directional movement: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Priority queue: (f_score, counter, (r, c))
        # counter breaks ties to avoid comparing coordinate tuples
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        counter = 0

        # g_score: exact cost from start to current cell
        g_score: dict = {start: 0}
        # came_from: tracks parent for path reconstruction
        came_from: dict = {}

        # Push start node
        heapq.heappush(open_set, (self._manhattan(start, end), counter, start))
        counter += 1

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for dr, dc in directions:
                nr, nc = current[0] + dr, current[1] + dc

                # Check bounds
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue

                # Skip walls
                if self.grid[nr][nc] == 0:
                    continue

                # Cost to enter neighbor cell
                move_cost = self.grid[nr][nc]
                tentative_g = g_score[current] + move_cost

                # Relaxation step
                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + self._manhattan((nr, nc), end)
                    heapq.heappush(open_set, (f_score, counter, (nr, nc)))
                    counter += 1

        return None

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two grid coordinates."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _reconstruct_path(came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Backtrack from end to start using parent pointers and reverse the path."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

import pytest

def test_basic_pathfinding():
    """Test standard pathfinding on an open grid."""
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
    assert len(path) == 5  # Shortest path length in 4-directional grid

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_no_path_blocked_by_walls():
    """Test that None is returned when the target is completely blocked."""
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
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (0, 0))

def test_start_or_end_on_wall_raises_value_error():
    """Test that placing start/end on a wall raises ValueError."""
    grid = [[1, 0], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 1), (1, 1))
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (0, 1))

def test_weighted_optimal_path():
    """Test that A* chooses the lowest-cost path, not necessarily the shortest Manhattan path."""
    # Center cell has high weight (10), surrounding cells have weight 1
    grid = [
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    # Optimal path should avoid the expensive center cell
    assert (1, 1) not in path
    
    # Verify total cost equals 5 (entering 5 cells of weight 1)
    cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert cost == 5