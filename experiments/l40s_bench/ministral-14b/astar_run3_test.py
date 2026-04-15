import heapq
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict

class AStarGrid:
    """A class to perform A* pathfinding on a weighted 2D grid."""

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid with movement costs.

        Args:
            grid: 2D list where each value represents movement cost (0 = impassable wall)
        """
        self.grid = grid
        self.rows = len(grid)
        if self.rows == 0:
            raise ValueError("Grid cannot be empty")
        self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm with Manhattan distance heuristic.

        Args:
            start: Starting coordinates (row, col)
            end: Target coordinates (row, col)

        Returns:
            List of coordinates representing the path from start to end (inclusive), or None if no path exists
        """
        # Validate coordinates
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start coordinates out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End coordinates out of bounds")

        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None

        # If start == end, return single point path
        if start == end:
            return [start]

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Priority queue: (f_score, g_score, current_node)
        open_set = []
        heapq.heappush(open_set, (0, 0, start))

        # For node n, came_from[n] is the node immediately preceding it on the cheapest path
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # For node n, g_score[n] is the cost of the cheapest path from start to n
        g_score: Dict[Tuple[int, int], int] = defaultdict(lambda: float('inf'))
        g_score[start] = 0

        # For node n, f_score[n] = g_score[n] + h(n)
        f_score: Dict[Tuple[int, int], int] = defaultdict(lambda: float('inf'))
        f_score[start] = self._manhattan_distance(start, end)

        open_set_hash: Set[Tuple[int, int]] = {start}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            open_set_hash.remove(current)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)

                # Skip if neighbor is out of bounds or a wall
                if (not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols) or
                    self.grid[neighbor[0]][neighbor[1]] == 0):
                    continue

                # Calculate tentative g_score
                tentative_g = current_g + self.grid[neighbor[0]][neighbor[1]]

                if tentative_g < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._manhattan_distance(neighbor, end)

                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
                        open_set_hash.add(neighbor)

        return None  # No path found

    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]],
                          current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get start to end

import pytest
from typing import List, Tuple, Optional

def test_simple_path_uniform_grid():
    """Test simple path on uniform grid (all cells have same cost)."""
    grid = [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ]
    solver = AStarGrid(grid)
    path = solver.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert sum(grid[r][c] for r, c in path) == 5  # Optimal cost

def test_path_around_obstacles():
    """Test path that goes around obstacles."""
    grid = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1]
    ]
    solver = AStarGrid(grid)
    path = solver.find_path((0, 0), (3, 4))
    assert path == [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 4), (2, 4), (3, 4), (3, 3), (3, 2), (3, 1), (3, 0), (2, 0), (1, 0), (0, 0)]
    # Actually, the optimal path should be around the obstacles
    # Let's check the actual optimal path
    optimal_path = [(0,0), (1,0), (2,0), (3,0), (3,1), (3,2), (3,3), (3,4)]
    assert solver.find_path((0, 0), (3, 4)) == optimal_path
    assert sum(grid[r][c] for r, c in optimal_path) == 8  # Optimal cost

def test_weighted_grid():
    """Test path prefers lower-cost cells."""
    grid = [
        [1, 2, 1],
        [1, 1, 1],
        [1, 2, 1]
    ]
    solver = AStarGrid(grid)
    path = solver.find_path((0, 0), (2, 2))
    # Should prefer the path through the 1-cost cells
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert sum(grid[r][c] for r, c in path) == 5  # Optimal cost

def test_no_path_exists():
    """Test when no path exists (fully blocked)."""
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    solver = AStarGrid(grid)
    assert solver.find_path((0, 0), (2, 2)) is None

def test_start_equals_end():
    """Test when start and end are the same."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    solver = AStarGrid(grid)
    assert solver.find_path((1, 1), (1, 1)) == [(1, 1)]

def test_invalid_coordinates():
    """Test invalid coordinates raise ValueError."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    solver = AStarGrid(grid)

    with pytest.raises(ValueError):
        solver.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        solver.find_path((0, 0), (3, 1))
    with pytest.raises(ValueError):
        solver.find_path((0, 0), (1, 3))

def test_start_or_end_is_wall():
    """Test when start or end is a wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    solver = AStarGrid(grid)

    assert solver.find_path((1, 1), (2, 2)) is None  # Start is wall
    assert solver.find_path((0, 0), (1, 1)) is None  # End is wall