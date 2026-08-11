```python
from typing import List, Tuple, Optional
import heapq


class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.

    Grid values represent movement cost (0 = impassable wall, positive int = cost to enter that cell).
    Supports 4-directional movement (up, down, left, right) with Manhattan distance heuristic.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid for pathfinding.

        Args:
            grid: 2D list where grid[row][col] is the movement cost (0 = wall, positive = cost)
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, row: int, col: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _is_wall(self, row: int, col: int) -> bool:
        """Check if a cell is a wall (cost 0)."""
        return self.grid[row][col] == 0

    def _manhattan_distance(self, row1: int, col1: int, row2: int, col2: int) -> int:
        """Calculate Manhattan distance between two cells."""
        return abs(row1 - row2) + abs(col1 - col2)

    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid 4-directional neighbors of a cell."""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self._is_valid(new_row, new_col) and not self._is_wall(new_row, new_col):
                neighbors.append((new_row, new_col))
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path from start to end using A* algorithm.

        Args:
            start: (row, col) starting coordinates
            end: (row, col) target coordinates

        Returns:
            List of (row, col) coordinates from start to end inclusive, or None if no path exists.

        Raises:
            ValueError: If start or end coordinates are out of bounds
        """
        # Validate coordinates
        if not self._is_valid(*start):
            raise ValueError(f"Start coordinates {start} are out of bounds")
        if not self._is_valid(*end):
            raise ValueError(f"End coordinates {end} are out of bounds")

        # Check if start or end are walls
        if self._is_wall(*start):
            return None
        if self._is_wall(*end):
            return None

        # Handle start == end case
        if start == end:
            return [start]

        # Initialize open set with start node
        # Each entry: (f_score, g_score, row, col, parent_row, parent_col)
        start_row, start_col = start
        end_row, end_col = end

        # g_score: cost from start to node
        g_score = {start: 0}
        # f_score = g_score + heuristic
        f_score = {start: self._manhattan_distance(start_row, start_col, end_row, end_col)}

        # Priority queue for open set
        open_heap = [(f_score[start], start_row, start_col, start_row, start_col)]

        # Track visited nodes and their parents
        came_from = {}
        visited = set()

        while open_heap:
            current_f, current_row, current_col, parent_row, parent_col = heapq.heappop(open_heap)
            current = (current_row, current_col)

            # Skip if already visited with better score
            if current in visited:
                continue

            visited.add(current)
            came_from[current] = (parent_row, parent_col)

            # Check if we reached the end
            if current == end:
                # Reconstruct path
                path = []
                node = current
                while node != start:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                path.reverse()
                return path

            # Explore neighbors
            for neighbor_row, neighbor_col in self._get_neighbors(current_row, current_col):
                neighbor = (neighbor_row, neighbor_col)

                # Calculate tentative g_score for neighbor
                tentative_g = g_score[current] + self.grid[neighbor_row][neighbor_col]

                # If neighbor not visited or better path found
                if neighbor not in visited or tentative_g < g_score.get(neighbor, float('inf')):
                    # Update scores
                    g_score[neighbor] = tentative_g
                    h_score = self._manhattan_distance(neighbor_row, neighbor_col, end_row, end_col)
                    f_score[neighbor] = tentative_g + h_score

                    # Add to open heap
                    heapq.heappush(open_heap, (f_score[neighbor], neighbor_row, neighbor_col, current_row, current_col))

        # No path found
        return None


# Test cases
import pytest

def test_simple_path_uniform_grid():
    """Test pathfinding on a uniform grid with no obstacles."""
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
    # Path should be optimal (minimum cost)
    # Each step costs 1, so total cost = len(path) - 1
    assert len(path) == 5  # (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) or similar
    # Verify path validity
    for i in range(len(path) - 1):
        row1, col1 = path[i]
        row2, col2 = path[i + 1]
        assert abs(row1 - row2) + abs(col1 - col2) == 1  # 4-directional movement

def test_path_around_obstacles():
    """Test pathfinding around obstacles."""
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 3))

    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 3)
    # Path must go around the wall in the middle
    # Verify no path goes through wall cells
    for row, col in path:
        assert grid[row][col] != 0  # No wall cells in path
    # Verify path is optimal (minimum cost)
    # All cells cost 1, so optimal path length should be minimal
    assert len(path) == 6  # Must go around the 2x2 wall block

def test_weighted_grid_prefers_lower_cost():
    """Test that pathfinding prefers lower-cost cells."""
    grid = [
        [1, 10, 10, 10],
        [1, 10, 10, 10],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 3))

    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 3)

    # Calculate total cost of path
    total_cost = sum(grid[row][col] for row, col in path[1:])  # Exclude start (no cost to enter)

    # Optimal path should go down through the low-cost cells (cost 1 each)
    # Path: (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2) -> (2,3)
    # Total cost = 1 + 1 + 1 + 1 + 1 = 5
    assert total_cost == 5

    # Verify path uses low-cost cells
    for row, col in path:
        assert grid[row][col] == 1  # All cells in path should be cost 1

def test_no_path_exists_fully_blocked():
    """Test when no path exists due to complete blockage."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))

    assert path is None  # No path around the single wall

def test_start_equals_end():
    """Test when start equals end."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))

    assert path == [(1, 1)]  # Single cell path

def test_invalid_coordinates():
    """Test invalid coordinates raise ValueError."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)

    # Out of bounds start
    with pytest.raises(ValueError):
        astar.find_path((5, 5), (0, 0))

    # Out of bounds end
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))

    # Both out of bounds
    with pytest.raises(ValueError):
        astar.find_path((5, 5), (5, 5))

def test_start_or_end_is_wall():
    """Test when start or end is a wall."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)

    # Start is wall
    assert astar.find_path((0, 0), (1, 1)) is None

    # End is wall
    assert astar.find_path((1, 1), (0, 0)) is None

    # Both are walls
    assert astar.find_path((0, 0), (0, 0)) is None
```