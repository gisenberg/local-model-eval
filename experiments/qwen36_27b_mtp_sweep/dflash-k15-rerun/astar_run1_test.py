import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid representation:
    - 0: Wall (impassable)
    - Positive integers: Weight/cost to enter the cell
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the AStarGrid with a 2D grid of weights.

        Args:
            grid: 2D list of integers. 0 represents a wall, positive integers 
                  represent traversal costs.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using the A* algorithm.

        Args:
            start: (row, col) tuple for the starting position.
            end: (row, col) tuple for the target position.

        Returns:
            A list of (row, col) tuples representing the optimal path from start to end,
            or None if no valid path exists.

        Raises:
            ValueError: If start or end is out of bounds or located on a wall.
        """
        # Validate coordinates
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds.")
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall.")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall.")

        if start == end:
            return [start]

        # Priority queue: (f_score, counter, position)
        # Counter ensures stable ordering when f_scores are equal
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))
        counter = 1

        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_set:
            f, _, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            # 4-directional movement: up, down, left, right
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)

                # Boundary and wall checks
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr][nc] == 0:
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

    @staticmethod
    def _heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _reconstruct_path(came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to end using the came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

import pytest

def test_basic_path():
    """Test pathfinding around a wall with a unique optimal route."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]

def test_start_equals_end():
    """Test immediate return when start and end are the same."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((1, 1), (1, 1)) == [(1, 1)]

def test_no_path():
    """Test return None when target is completely blocked by walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None

def test_out_of_bounds():
    """Test ValueError raised for coordinates outside grid dimensions."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_at_start_or_end():
    """Test ValueError raised when start or end position is a wall."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((1, 1), (0, 0))

def test_weighted_optimality():
    """Test that A* chooses lower-weight paths over shorter geometric paths."""
    grid = [
        [1, 10, 10],
        [1,  1,  1],
        [1,  1,  1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    # Optimal path goes down first to avoid weight-10 cells
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    
    # Verify total cost is minimal (4 instead of 10+)
    cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert cost == 4