import heapq
from typing import List, Tuple, Optional, Set

class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.

    Attributes:
        grid (List[List[int]]): 2D grid where 0 represents walls and positive integers represent weights.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a given grid.

        Args:
            grid: 2D list representing the grid. 0 = wall, positive int = weight.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance heuristic.

        Args:
            pos: Current position (row, col)
            end: Target position (row, col)

        Returns:
            Manhattan distance between pos and end
        """
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    def _is_valid(self, row: int, col: int) -> bool:
        """
        Check if a position is within bounds and not a wall.

        Args:
            row: Row index
            col: Column index

        Returns:
            True if position is valid, False otherwise
        """
        return (0 <= row < self.rows and
                0 <= col < self.cols and
                self.grid[row][col] != 0)

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.

        Args:
            start: Starting position (row, col)
            end: Target position (row, col)

        Returns:
            List of positions forming the path, or None if no path exists

        Raises:
            ValueError: If start or end is out of bounds or is a wall
        """
        # Validate inputs
        if not self._is_valid(start[0], start[1]):
            raise ValueError(f"Start position {start} is invalid")
        if not self._is_valid(end[0], end[1]):
            raise ValueError(f"End position {end} is invalid")

        # Handle case where start equals end
        if start == end:
            return [start]

        # Initialize data structures
        open_set = []  # Priority queue: (f_score, g_score, position)
        closed_set: Set[Tuple[int, int]] = set()

        # Start with the initial position
        start_g = 0
        start_f = self._heuristic(start, end)
        heapq.heappush(open_set, (start_f, start_g, start))

        # Track the best g_score for each position
        g_scores = {start: 0}
        # Track parent for path reconstruction
        parents = {start: None}

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            # Get the node with lowest f_score
            current_f, current_g, current = heapq.heappop(open_set)

            # Skip if already processed
            if current in closed_set:
                continue

            # Mark as processed
            closed_set.add(current)

            # Check if we've reached the goal
            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = parents[current]
                return path[::-1]  # Reverse to get start-to-end path

            # Explore neighbors
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)

                # Skip invalid positions
                if not self._is_valid(neighbor[0], neighbor[1]):
                    continue

                # Skip already processed nodes
                if neighbor in closed_set:
                    continue

                # Calculate tentative g_score
                neighbor_g = current_g + self.grid[neighbor[0]][neighbor[1]]

                # If this path to neighbor is better, update it
                if neighbor not in g_scores or neighbor_g < g_scores[neighbor]:
                    g_scores[neighbor] = neighbor_g
                    f_score = neighbor_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor_g, neighbor))
                    parents[neighbor] = current

        # No path found
        return None


# Tests
import pytest

def test_start_equals_end():
    """Test when start position equals end position."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (0, 0))
    assert result == [(0, 0)]

def test_simple_path():
    """Test finding a simple path in an open grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (2, 2))
    assert result is not None
    assert result[0] == (0, 0)
    assert result[-1] == (2, 2)
    # Should be 5 steps (4 moves)
    assert len(result) == 5

def test_wall_blocking():
    """Test when walls block all possible paths."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (0, 2))
    assert result is None

def test_weighted_grid():
    """Test pathfinding on a weighted grid chooses optimal path."""
    grid = [
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (2, 2))
    assert result is not None
    # Optimal path should avoid the high-weight cell at (1,1)
    assert (1, 1) not in result

def test_invalid_start():
    """Test that invalid start position raises ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))  # Out of bounds

def test_wall_start():
    """Test that starting on a wall raises ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))  # Start is a wall