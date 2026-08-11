import heapq
from typing import List, Optional, Tuple, Set
from collections import deque

class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.

    Attributes:
        grid (List[List[int]]): The 2D grid where 0 represents walls,
                               and positive integers represent movement costs.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D grid.

        Args:
            grid: 2D list where 0 represents walls and positive integers
                  represent movement costs.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0

    def _is_valid(self, row: int, col: int) -> bool:
        """Check if a position is within grid bounds and not a wall."""
        return (0 <= row < self.rows and
                0 <= col < self.cols and
                self.grid[row][col] != 0)

    def _heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid 4-directional neighbors."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        neighbors = []
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self._is_valid(new_row, new_col):
                neighbors.append((new_row, new_col))
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start: Starting position as (row, col)
            end: Ending position as (row, col)

        Returns:
            List of positions forming the path, or None if no path exists.

        Raises:
            ValueError: If start or end positions are out of bounds or on walls.
        """
        # Validate inputs
        if not self._is_valid(start[0], start[1]):
            raise ValueError(f"Start position {start} is out of bounds or on a wall")
        if not self._is_valid(end[0], end[1]):
            raise ValueError(f"End position {end} is out of bounds or on a wall")

        # Handle start == end case
        if start == end:
            return [start]

        # Initialize data structures
        open_set = [(0, start)]  # (f_score, position)
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, end)}

        while open_set:
            current_f, current = heapq.heappop(open_set)

            # Found the goal
            if current == end:
                return self._reconstruct_path(came_from, current)

            # Explore neighbors
            for neighbor in self._get_neighbors(current[0], current[1]):
                # Calculate tentative g_score
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                # If this path is better, update
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        # No path found
        return None

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from start to end."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get path from start to end

import pytest
from typing import List, Tuple, Optional

# Assuming the AStarGrid class is defined above

def test_start_equals_end():
    """Test when start and end positions are the same."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    result = astar.find_path((1, 1), (1, 1))
    assert result == [(1, 1)]

def test_simple_path():
    """Test a simple path without obstacles."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (2, 2))
    assert result is not None
    assert result[0] == (0, 0)
    assert result[-1] == (2, 2)
    # Check that path length is correct (minimum steps)
    assert len(result) == 5  # 4 moves + start

def test_wall_obstacle():
    """Test pathfinding around a wall obstacle."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (0, 2))
    assert result is not None
    assert result[0] == (0, 0)
    assert result[-1] == (0, 2)
    # Path should go around the wall
    assert len(result) == 6  # Going around the wall

def test_no_path():
    """Test when no path exists due to complete blockage."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (2, 2))
    assert result is None

def test_weighted_cells():
    """Test pathfinding with weighted cells."""
    grid = [
        [1, 1, 1],
        [1, 9, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (2, 2))
    assert result is not None
    assert result[0] == (0, 0)
    assert result[-1] == (2, 2)
    # Path should avoid the expensive cell (1,1) with cost 9
    assert (1, 1) not in result

def test_out_of_bounds():
    """Test that out of bounds raises ValueError."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)

    # Test out of bounds start
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (2, 2))

    with pytest.raises(ValueError):
        astar.find_path((0, 5), (2, 2))

    # Test out of bounds end
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))

    # Test wall positions
    grid_with_walls = [
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 0]
    ]
    astar_walls = AStarGrid(grid_with_walls)

    with pytest.raises(ValueError):
        astar_walls.find_path((0, 0), (2, 2))  # Start on wall

    with pytest.raises(ValueError):
        astar_walls.find_path((0, 1), (2, 2))  # End on wall