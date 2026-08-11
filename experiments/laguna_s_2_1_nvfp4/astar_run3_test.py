import heapq
from typing import List, Tuple, Optional, Set, Dict

class AStarGrid:
    """
    A* pathfinding algorithm implementation on a weighted 2D grid.

    Attributes:
        grid (List[List[int]]): 2D grid where 0 represents walls and positive integers represent weights.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D grid.

        Args:
            grid: 2D list representing the grid. 0 = wall, positive int = weight.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, row: int, col: int) -> bool:
        """Check if a position is within grid bounds."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid 4-directional neighbors that are not walls."""
        row, col = pos
        neighbors = []
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # right, down, left, up

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if (self._is_valid(new_row, new_col) and
                self.grid[new_row][new_col] != 0):  # Not a wall
                neighbors.append((new_row, new_col))

        return neighbors

    def _get_weight(self, pos: Tuple[int, int]) -> int:
        """Get the weight of a cell."""
        return self.grid[pos[0]][pos[1]]

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start: Starting position as (row, col).
            end: Ending position as (row, col).

        Returns:
            List of positions forming the path, or None if no path exists.

        Raises:
            ValueError: If start or end positions are out of bounds or are walls.
        """
        # Validate inputs
        if not self._is_valid(start[0], start[1]):
            raise ValueError(f"Start position {start} is out of bounds")
        if not self._is_valid(end[0], end[1]):
            raise ValueError(f"End position {end} is out of bounds")
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError(f"Start position {start} is a wall")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError(f"End position {end} is a wall")

        # Special case: start == end
        if start == end:
            return [start]

        # Initialize A* data structures
        open_set: List[Tuple[int, int, int, int]] = []  # (f_score, g_score, row, col)
        heapq.heappush(open_set, (0, 0, start[0], start[1]))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}

        # Closed set to track visited nodes
        closed_set: Set[Tuple[int, int]] = set()

        while open_set:
            current_f, current_g, current_row, current_col = heapq.heappop(open_set)
            current = (current_row, current_col)

            # Skip if already processed
            if current in closed_set:
                continue

            closed_set.add(current)

            # Check if we've reached the goal
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue

                # Calculate tentative g_score
                tentative_g = current_g + self._get_weight(neighbor)

                # If this path to neighbor is better, update
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor[0], neighbor[1]))

        # No path found
        return None


# Pytest tests
import pytest

def test_basic_path():
    """Test basic pathfinding on a simple grid."""
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
    assert len(path) == 5  # Manhattan distance + 1

def test_start_equals_end():
    """Test when start and end are the same."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_wall_blocking():
    """Test pathfinding when walls block the direct path."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (0, 2)
    # Should go around the wall
    assert len(path) == 7  # Must detour

def test_no_path():
    """Test when no path exists due to complete wall blockage."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_weighted_path():
    """Test that A* chooses optimal weighted path."""
    grid = [
        [1, 10, 10, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 3))
    assert path is not None
    # Should prefer the bottom path (cost 4) over top path (cost 21)
    assert len(path) == 7  # Goes down, right x3, then up

def test_out_of_bounds():
    """Test that out-of-bounds positions raise ValueError."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 2))

def test_wall_positions():
    """Test that starting/ending on walls raises ValueError."""
    grid = [
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 1), (1, 2))  # Start on wall
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 1))  # End on wall