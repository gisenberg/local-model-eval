import heapq
from typing import List, Tuple, Optional, Set
import pytest


class AStarGrid:
    """
    A* pathfinding algorithm for a weighted 2D grid.

    Attributes:
        grid: 2D list representing the grid where 0 indicates a wall and other values indicate weights.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the AStarGrid with a given grid.

        Args:
            grid: 2D list of integers representing the grid.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from start to end using A* algorithm.

        Args:
            start: Starting position as (row, col).
            end: Ending position as (row, col).

        Returns:
            List of positions representing the path, or None if no path exists.

        Raises:
            ValueError: If start or end is out of bounds or is a wall.
        """
        # Validate inputs
        if not self._is_valid_position(start):
            raise ValueError(f"Start position {start} is out of bounds or a wall")
        if not self._is_valid_position(end):
            raise ValueError(f"End position {end} is out of bounds or a wall")

        # Handle case where start equals end
        if start == end:
            return [start]

        # Initialize data structures
        open_set: List[Tuple[int, int, int, int]] = []  # (f_score, g_score, row, col)
        closed_set: Set[Tuple[int, int]] = set()
        came_from: dict = {}

        # g_score: cost from start to current node
        g_score = {start: 0}

        # f_score: g_score + heuristic
        f_score = {start: self._heuristic(start, end)}

        # Add start to open set
        heapq.heappush(open_set, (f_score[start], 0, start[0], start[1]))

        while open_set:
            # Get node with lowest f_score
            current_f, current_g, current_row, current_col = heapq.heappop(open_set)
            current = (current_row, current_col)

            # Check if we've reached the end
            if current == end:
                return self._reconstruct_path(came_from, current)

            # Skip if already processed
            if current in closed_set:
                continue

            # Mark as processed
            closed_set.add(current)

            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue

                # Calculate tentative g_score
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                # If this path to neighbor is better, or neighbor is not in open set
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)

                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor[0], neighbor[1]))

        # No path found
        return None

    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """
        Checks if a position is within bounds and not a wall.

        Args:
            pos: Position to check as (row, col).

        Returns:
            True if position is valid, False otherwise.
        """
        row, col = pos
        if row < 0 or row >= self.rows or col < 0 or col >= self.cols:
            return False
        return self.grid[row][col] != 0

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Gets valid neighboring positions (up, down, left, right).

        Args:
            pos: Current position as (row, col).

        Returns:
            List of valid neighbor positions.
        """
        row, col = pos
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right

        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols and self.grid[new_row][new_col] != 0:
                neighbors.append((new_row, new_col))

        return neighbors

    def _heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """
        Calculates Manhattan distance heuristic.

        Args:
            pos: Current position.
            end: Target position.

        Returns:
            Manhattan distance between pos and end.
        """
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstructs the path from start to end.

        Args:
            came_from: Dictionary mapping each node to its predecessor.
            current: End node.

        Returns:
            List of positions from start to end.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


# Tests
class TestAStarGrid:
    """Tests for AStarGrid class."""

    def test_simple_path(self):
        """Test finding a simple path in an empty grid."""
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
        # Verify path is valid (each step is adjacent)
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            assert abs(r1 - r2) + abs(c1 - c2) == 1

    def test_start_equals_end(self):
        """Test when start and end are the same."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((1, 1), (1, 1))
        assert path == [(1, 1)]

    def test_wall_blocking_path(self):
        """Test when walls block all paths."""
        grid = [
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        assert path is None

    def test_weighted_path(self):
        """Test that A* finds the optimal path considering weights."""
        # Grid where going around is cheaper than through high-weight cell
        grid = [
            [1, 1, 1, 1, 1],
            [1, 9, 9, 9, 1],
            [1, 1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 4))
        assert path is not None
        # Should go around the high-cost cells
        assert (1, 1) not in path

    def test_out_of_bounds_raises_error(self):
        """Test that out-of-bounds positions raise ValueError."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (2, 2))
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (3, 3))

    def test_wall_as_start_or_end_raises_error(self):
        """Test that using a wall as start or end raises ValueError."""
        grid = [
            [1, 0, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        with pytest.raises(ValueError):
            astar.find_path((0, 1), (2, 2))  # Start is a wall
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (0, 1))  # End is a wall