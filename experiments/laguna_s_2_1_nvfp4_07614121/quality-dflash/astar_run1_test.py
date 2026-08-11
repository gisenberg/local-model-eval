import heapq
from typing import List, Optional, Tuple, Set, Dict

class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.

    Grid values:
        0: wall (impassable)
        1: normal terrain (cost 1)
        >1: weighted terrain (cost = value)

    Movement: 4-directional (up, down, left, right)
    Heuristic: Manhattan distance
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.

        Args:
            grid: 2D list representing the grid with weights.
                  0 = wall, 1 = normal, >1 = weighted terrain.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")

        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

        # Validate grid dimensions
        for row in grid:
            if len(row) != self.cols:
                raise ValueError("All rows must have the same length")

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start: Starting position as (row, col)
            end: Ending position as (row, col)

        Returns:
            List of (row, col) tuples representing the path from start to end,
            or None if no path exists.

        Raises:
            ValueError: If start or end is out of bounds or on a wall.
        """
        # Validate inputs
        self._validate_position(start)
        self._validate_position(end)

        # Handle start == end case
        if start == end:
            return [start]

        # A* algorithm
        open_heap: List[Tuple[int, int, int, int]] = [(0, 0, start[0], start[1])]
        # (f_score, g_score, row, col)

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        g_score: Dict[Tuple[int, int], int] = {start: 0}
        closed_set: Set[Tuple[int, int]] = set()

        while open_heap:
            current_f, current_g, current_row, current_col = heapq.heappop(open_heap)
            current = (current_row, current_col)

            # Skip if already processed with better score
            if current in closed_set:
                continue

            # Check if we've reached the goal
            if current == end:
                return self._reconstruct_path(came_from, current)

            closed_set.add(current)

            # Explore neighbors
            for neighbor_row, neighbor_col in self._get_neighbors(current_row, current_col):
                neighbor = (neighbor_row, neighbor_col)

                if neighbor in closed_set:
                    continue

                # Calculate tentative g_score
                tentative_g = current_g + self.grid[neighbor_row][neighbor_col]

                # If this path to neighbor is better, update
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f_score, tentative_g, neighbor_row, neighbor_col))

        # No path found
        return None

    def _validate_position(self, pos: Tuple[int, int]) -> None:
        """Validate that a position is within bounds and not a wall."""
        row, col = pos
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"Position {pos} is out of bounds")
        if self.grid[row][col] == 0:
            raise ValueError(f"Position {pos} is a wall")

    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid 4-directional neighbors."""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (0 <= new_row < self.rows and
                0 <= new_col < self.cols and
                self.grid[new_row][new_col] != 0):  # Not a wall
                neighbors.append((new_row, new_col))

        return neighbors

    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]],
                         current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from start to current using came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


# Tests
import pytest

def test_simple_path():
    """Test finding a simple path in an open grid."""
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
    # Verify it's a valid path (each step moves by 1 in one direction)
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1

def test_start_equals_end():
    """Test when start equals end."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_due_to_walls():
    """Test when no path exists due to walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds():
    """Test that out of bounds positions raise ValueError."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)

    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (2, 2))

    with pytest.raises(ValueError):
        astar.find_path((0, 0), (3, 0))

def test_weighted_terrain_optimal_path():
    """Test that A* finds optimal path considering weights."""
    # Grid where going around is cheaper than through expensive cell
    grid = [
        [1, 1, 1],
        [1, 5, 1],  # Expensive middle cell
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Path should avoid the expensive cell (1,1)
    assert (1, 1) not in path

def test_wall_position_raises_error():
    """Test that starting/ending on a wall raises ValueError."""
    grid = [
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)

    with pytest.raises(ValueError):
        astar.find_path((0, 1), (2, 2))  # Start on wall

    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 1))  # End on wall