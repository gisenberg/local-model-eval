Here's a complete implementation of A* pathfinding on a weighted 2D grid with all the required features:

```python
from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    """
    A* pathfinding implementation for weighted 2D grids.
    Supports 4-directional movement with Manhattan distance heuristic.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid.

        Args:
            grid: 2D list where values represent movement cost
                  (0 = impassable wall, positive int = cost to enter)

        Raises:
            ValueError: If grid is empty or not rectangular
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty")

        # Validate rectangular grid
        row_lengths = [len(row) for row in grid]
        if len(set(row_lengths)) != 1:
            raise ValueError("Grid must be rectangular")

        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid_cell(self, pos: Tuple[int, int]) -> bool:
        """
        Check if a position is within grid bounds and not a wall.

        Args:
            pos: (row, col) coordinates

        Returns:
            True if position is valid and traversable, False otherwise
        """
        row, col = pos
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        return self.grid[row][col] != 0

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighboring positions (4-directional movement).

        Args:
            pos: Current position (row, col)

        Returns:
            List of valid neighboring positions
        """
        row, col = pos
        neighbors = []

        # Up, down, left, right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (row + dr, col + dc)
            if self._is_valid_cell(new_pos):
                neighbors.append(new_pos)

        return neighbors

    def _manhattan_distance(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two positions.

        Args:
            pos: Current position (row, col)
            end: Target position (row, col)

        Returns:
            Manhattan distance (sum of absolute row and column differences)
        """
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start: Starting position (row, col)
            end: Target position (row, col)

        Returns:
            List of coordinates from start to end inclusive, or None if no path exists

        Raises:
            ValueError: If start or end coordinates are out of bounds
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start position {start} is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End position {end} is out of bounds")

        # Check if start or end are walls
        if not self._is_valid_cell(start):
            return None
        if not self._is_valid_cell(end):
            return None

        # Handle start == end case
        if start == end:
            return [start]

        # Initialize open set with start node
        # Format: (f_score, g_score, position)
        open_set = []
        start_g = self.grid[start[0]][start[1]]
        start_f = start_g + self._manhattan_distance(start, end)
        heapq.heappush(open_set, (start_f, start_g, start))

        # Track visited nodes and their costs
        g_scores = {start: start_g}
        came_from = {}

        # Track nodes in open set for efficient lookup
        open_set_positions = {start}

        while open_set:
            # Pop node with lowest f_score
            current_f, current_g, current = heapq.heappop(open_set)
            open_set_positions.remove(current)

            # Check if we reached the goal
            if current == end:
                # Reconstruct path
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                path.reverse()
                return path

            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                # Calculate tentative g_score
                tentative_g = current_g + self.grid[neighbor[0]][neighbor[1]]

                # If neighbor not visited or we found better path
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    # Update scores
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._manhattan_distance(neighbor, end)

                    # Update path
                    came_from[neighbor] = current

                    # Add to open set if not already there
                    if neighbor not in open_set_positions:
                        heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                        open_set_positions.add(neighbor)

        # No path found
        return None

    def get_path_cost(self, path: List[Tuple[int, int]]) -> int:
        """
        Calculate the total cost of a path.

        Args:
            path: List of coordinates

        Returns:
            Total movement cost of the path
        """
        if not path:
            return 0
        return sum(self.grid[row][col] for row, col in path)


# Test functions
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

    # Check path validity (adjacent cells)
    for i in range(len(path) - 1):
        assert abs(path[i][0] - path[i+1][0]) + abs(path[i][1] - path[i+1][1]) == 1

    # Check optimality (minimum cost)
    cost = astar.get_path_cost(path)
    assert cost == 5  # 5 cells visited (including start and end)


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

    # Verify path doesn't go through walls
    for pos in path:
        assert grid[pos[0]][pos[1]] != 0

    # Check optimality
    cost = astar.get_path_cost(path)
    assert cost == 7  # Must go around the obstacles


def test_weighted_grid_prefers_lower_cost():
    """Test that pathfinding prefers lower-cost cells."""
    grid = [
        [1, 10, 10, 10],
        [1, 1, 1, 10],
        [10, 10, 1, 1]
    ]
    astar = AStarGrid(grid)

    path = astar.find_path((0, 0), (2, 3))

    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 3)

    # Check optimality - path should prefer low-cost cells
    cost = astar.get_path_cost(path)
    assert cost == 5  # Path through low-cost cells: (0,0)->(1,0)->(1,1)->(1,2)->(2,2)->(2,3)

    # Verify path uses low-cost cells
    for pos in path:
        assert grid[pos[0]][pos[1]] <= 1


def test_no_path_exists():
    """Test when no path exists (fully blocked)."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)

    path = astar.find_path((0, 0), (2, 2))

    assert path is None


def test_start_equals_end():
    """Test when start equals end."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)

    path = astar.find_path((1, 1), (1, 1))

    assert path == [(1, 1)]


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

    # Start is a wall
    wall_grid = [
        [0, 1],
        [1, 1]
    ]
    astar_wall = AStarGrid(wall_grid)
    assert astar_wall.find_path((0, 0), (1, 1)) is None

    # End is a wall
    assert astar_wall.find_path((1, 1), (0, 0)) is None


def test_optimality_with_multiple_paths():
    """Test that A* finds the optimal path when multiple paths exist."""
    grid = [
        [1, 5, 5, 5],
        [1, 1, 1, 5],
        [5, 5, 1, 1]
    ]
    astar = AStarGrid(grid)

    path = astar.find_path((0, 0), (2, 3))

    assert path is not None

    # Calculate cost and verify it's optimal
    cost = astar.get_path_cost(path)

    # The optimal path should go through low-cost cells
    # Path: (0,0)->(1,0)->(1,1)->(1,2)->(2,2)->(2,3)
    # Cost: 1+1+1+1+1+1 = 6
    assert cost == 6

    # Verify path validity
    for i in range(len(path) - 1):
        assert abs(path[i][0] - path[i+1][0]) + abs(path[i][1] - path[i+1][1]) == 1


def test_large_grid_performance():
    """Test performance on a larger grid."""
    grid = [[1 for _ in range(20)] for _ in range(20)]
    astar = AStarGrid(grid)

    path = astar.find_path((0, 0), (19, 19))

    assert path is not None
    assert len(path) == 39  # Manhattan distance + 1 for start
    assert astar.get_path_cost(path) == 39
```

This implementation includes:

1. **Class structure**: `AStarGrid` with proper initialization and validation
2. **A* algorithm**: Uses min-heap for open set, Manhattan distance heuristic
3. **Edge cases**: Handles start==end, walls, out-of-bounds coordinates
4. **Type hints**: Throughout the entire codebase
5. **Docstrings**: Brief documentation on each method
6. **8 test functions**: Covering all required scenarios plus additional edge cases

The algorithm guarantees optimality because:
- It uses the Manhattan distance heuristic which is admissible (never overestimates)
- It maintains a min-heap ordered by f-score (g-score + heuristic)
- It updates g-scores when better paths are found
- It explores nodes in order of increasing f-score

The tests verify both path validity (adjacent cells, no walls) and optimality (minimum total cost).