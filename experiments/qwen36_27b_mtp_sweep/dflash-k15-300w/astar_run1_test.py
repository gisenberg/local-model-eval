import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """A* pathfinding algorithm on a weighted 2D grid."""

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid for pathfinding.
        
        :param grid: 2D list where 0 represents walls (impassable) and 
                     positive integers represent the cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_in_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within grid boundaries."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using the A* algorithm.
        
        :param start: (row, col) tuple representing the starting position.
        :param end: (row, col) tuple representing the target position.
        :return: List of (row, col) tuples representing the optimal path, 
                 or None if no valid path exists.
        :raises ValueError: If start or end is out of bounds or positioned on a wall.
        """
        # Validate bounds
        if not self._is_in_bounds(start) or not self._is_in_bounds(end):
            raise ValueError("Start or end position is out of bounds.")
        
        # Validate walls
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            raise ValueError("Start or end position is on a wall.")

        # Handle trivial case
        if start == end:
            return [start]

        # Priority queue: (f_score, tie_breaker, position)
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))

        # g_score tracks the lowest cost from start to each node
        g_score: dict[Tuple[int, int], float] = {start: 0}
        # came_from tracks the path reconstruction
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        
        counter = 1  # Tie-breaker to prevent comparing tuples in heapq

        while open_set:
            f, _, current = heapq.heappop(open_set)

            # Goal reached
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Skip stale entries
            if f > g_score.get(current, float('inf')):
                continue

            r, c = current
            # 4-directional movement
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)

                if not self._is_in_bounds(neighbor) or self.grid[nr][nc] == 0:
                    continue

                # Cost to move into the neighbor cell
                move_cost = self.grid[nr][nc]
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    # Manhattan heuristic
                    h = abs(nr - end[0]) + abs(nc - end[1])
                    f_score = tentative_g + h
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1

        return None

import pytest

def test_basic_pathfinding():
    """Test standard pathfinding on an unweighted grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Optimal path length in 3x3 grid

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1]]
    path = AStarGrid(grid).find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_no_path_due_to_walls():
    """Test that None is returned when walls completely block the path."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    assert AStarGrid(grid).find_path((0, 0), (0, 2)) is None

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_at_start_or_end_raises_value_error():
    """Test that starting or ending on a wall raises ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="on a wall"):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError, match="on a wall"):
        astar.find_path((1, 1), (0, 0))

def test_weighted_optimal_path():
    """Test that A* chooses the lowest-cost path over the shortest Manhattan path."""
    grid = [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 10, 1, 1],
        [1, 1, 1, 1]
    ]
    path = AStarGrid(grid).find_path((0, 0), (3, 3))
    assert path is not None
    # The algorithm should route around the expensive cell at (2, 1)
    assert (2, 1) not in path
    # Verify total cost is optimal (4 steps * cost 1 = 4)
    total_cost = sum(grid[r][c] for r, c in path[1:])  # Exclude start cell cost
    assert total_cost == 4