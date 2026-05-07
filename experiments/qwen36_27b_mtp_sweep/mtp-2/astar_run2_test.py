import heapq
from typing import List, Tuple, Optional
import pytest


class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid representation:
    - 0: Wall (impassable)
    - Positive integers: Movement cost/weight to enter the cell
    - Coordinates are (row, col) tuples
    """
    def __init__(self, grid: List[List[int]]) -> None:
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not a wall."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic for 4-directional movement."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: (row, col) tuple of starting position
            end: (row, col) tuple of ending position
            
        Returns:
            List of (row, col) tuples representing the optimal path, 
            or None if no valid path exists.
            
        Raises:
            ValueError: If start or end is out of bounds or placed on a wall.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds or on a wall.")

        if start == end:
            return [start]

        # Priority queue stores: (f_score, tie_breaker_counter, position)
        open_set = []
        counter = 0
        heapq.heappush(open_set, (self._heuristic(start, end), counter, start))
        counter += 1

        g_score = {start: 0}
        came_from = {}
        closed_set = set()

        while open_set:
            f, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            r, c = current
            # 4-directional movement: right, left, down, up
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)

                if neighbor in closed_set or not self._is_valid(neighbor):
                    continue

                weight = self.grid[nr][nc]
                tentative_g = g_score[current] + weight

                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1

        return None


# ========================
# Pytest Test Suite
# ========================

def test_basic_straight_path():
    """Test simple pathfinding on an open grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Optimal Manhattan path length

def test_path_around_walls():
    """Test that path correctly navigates around walls."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert (1, 1) not in path  # Wall should not be in path

def test_start_equals_end():
    """Test immediate return when start and end are the same."""
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_no_path_exists():
    """Test return None when destination is completely blocked."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
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

def test_weighted_optimal_path():
    """Test that A* chooses lower-cost path over shorter Manhattan path."""
    # Direct path cost: 10 + 1 = 11
    # Detour path cost: 1+1+1+1+1+1 = 6
    grid = [
        [1, 10, 1],
        [1,  1, 1],
        [1,  1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    # First step should go down to avoid high-weight cells
    assert path[1] == (1, 0)
    # Verify total cost matches optimal
    total_cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert total_cost == 6