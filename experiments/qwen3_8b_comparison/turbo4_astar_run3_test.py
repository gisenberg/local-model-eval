from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    """
    A class to represent a weighted 2D grid for A* pathfinding.
    Grid values represent movement cost (0 = impassable wall, positive int = cost to enter that cell).
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid with movement costs.

        Args:
            grid (List[List[int]]): 2D list of integers representing movement costs.
        """
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0]) if self.height > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.

        Args:
            start (Tuple[int, int]): Starting position (row, col)
            end (Tuple[int, int]): Target position (row, col)

        Returns:
            Optional[List[Tuple[int, int]]: List of coordinates representing the path, or None if no path exists.
        """
        # Check for invalid inputs
        if not self._is_valid_position(start) or not self._is_valid_position(end):
            raise ValueError("Start or end position is out of bounds.")
        if start == end:
            return [start]

        # Initialize open set and closed set
        open_set = [(0, start)]  # (f_score, position)
        came_from = {}  # Maps positions to their predecessors
        g_score = {start: 0}  # Cost from start to each position
        f_score = {start: self._heuristic(start, end)}  # f = g + h

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == end:
                # Reconstruct the path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            # Skip if current is already processed
            if current in closed_set:
                continue

            # Generate neighbors
            neighbors = self._get_neighbors(current)
            for neighbor in neighbors:
                tentative_g_score = g_score[current] + self._get_cost(current, neighbor)

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

            # Add current to closed set
            closed_set.add(current)

        # No path found
        return None

    def _get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get all valid 4-directional neighbors of a position.

        Args:
            position (Tuple[int, int]): Position (row, col)

        Returns:
            List[Tuple[int, int]]: List of valid neighbor positions.
        """
        row, col = position
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if self._is_valid_position((new_row, new_col)):
                neighbors.append((new_row, new_col))
        return neighbors

    def _get_cost(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
        """
        Get the movement cost from from_pos to to_pos.

        Args:
            from_pos (Tuple[int, int]): Starting position
            to_pos (Tuple[int, int]): Target position

        Returns:
            int: Movement cost from from_pos to to_pos
        """
        return self.grid[to_pos[0]][to_pos[1]]

    def _heuristic(self, position: Tuple[int, int], end: Tuple[int, int]) -> int:
        """
        Manhattan heuristic: distance from position to end.

        Args:
            position (Tuple[int, int]): Current position
            end (Tuple[int, int]): Target position

        Returns:
            int: Heuristic estimate of the cost from position to end
        """
        return abs(position[0] - end[0]) + abs(position[1] - end[1])

    def _is_valid_position(self, position: Tuple[int, int]) -> bool:
        """
        Check if a position is within bounds and not a wall.

        Args:
            position (Tuple[int, int]): Position (row, col)

        Returns:
            bool: True if position is valid, False otherwise
        """
        row, col = position
        return 0 <= row < self.height and 0 <= col < self.width and self.grid[row][col] != 0

import pytest
from typing import List, Tuple, Optional

# Test 1: Basic path with no walls
def test_basic_path():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    grid_obj = AStarGrid(grid)
    path = grid_obj.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

# Test 2: Start == end
def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    grid_obj = AStarGrid(grid)
    path = grid_obj.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

# Test 3: Wall in the way
def test_wall_in_way():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    grid_obj = AStarGrid(grid)
    path = grid_obj.find_path((0, 0), (2, 2))
    assert path is None

# Test 4: Out-of-bounds start
def test_out_of_bounds_start():
    grid = [[1, 1], [1, 1]]
    grid_obj = AStarGrid(grid)
    with pytest.raises(ValueError):
        grid_obj.find_path((3, 0), (0, 0))

# Test 5: Out-of-bounds end
def test_out_of_bounds_end():
    grid = [[1, 1], [1, 1]]
    grid_obj = AStarGrid(grid)
    with pytest.raises(ValueError):
        grid_obj.find_path((0, 0), (3, 0))

# Test 6: Path with varying costs
def test_path_with_varying_costs():
    grid = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
    grid_obj = AStarGrid(grid)
    path = grid_obj.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]