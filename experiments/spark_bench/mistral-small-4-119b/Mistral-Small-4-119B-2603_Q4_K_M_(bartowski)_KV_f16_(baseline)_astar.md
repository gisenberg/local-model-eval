# Mistral-Small-4-119B-2603 Q4_K_M (bartowski) — KV f16 (baseline) — A* Pathfinding

## Output

# A* Pathfinding on a Weighted 2D Grid

Here's an implementation of A* pathfinding on a weighted 2D grid with all the requested requirements:

```python
import heapq
from typing import List, Tuple, Optional, Dict, Set

class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.

    The grid values represent movement costs (0 = impassable wall, positive int = cost to enter that cell).
    Supports 4-directional movement (up, down, left, right).
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid with movement costs.

        Args:
            grid: A 2D list where each element represents the movement cost of that cell.
                 0 indicates an impassable wall.

        Raises:
            ValueError: If the grid is empty or rows have inconsistent lengths.
        """
        if not grid or not all(len(row) == len(grid[0]) for row in grid):
            raise ValueError("Grid must be non-empty and have consistent row lengths")

        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start: Tuple of (row, col) coordinates for the starting position.
            end: Tuple of (row, col) coordinates for the target position.

        Returns:
            List of (row, col) coordinates representing the path from start to end (inclusive),
            or None if no path exists.

        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        if not self._is_valid_coordinate(start) or not self._is_valid_coordinate(end):
            raise ValueError("Start or end coordinates are out of bounds")

        if start == end:
            return [start]

        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Priority queue for open set: (f_score, g_score, current_node, path)
        open_set = []
        heapq.heappush(open_set, (0, 0, start, [start]))

        # For tracking the best path to each node
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # Cost from start to each node
        g_score: Dict[Tuple[int, int], float] = {start: 0}

        # Set of visited nodes
        visited: Set[Tuple[int, int]] = set()

        while open_set:
            _, current_g, current_node, path = heapq.heappop(open_set)

            if current_node == end:
                return path

            if current_node in visited:
                continue

            visited.add(current_node)

            for neighbor in self._get_neighbors(current_node):
                if neighbor in visited:
                    continue

                tentative_g = current_g + self.grid[neighbor[0]][neighbor[1]]

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._manhattan_distance(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, path + [neighbor]))
                    came_from[neighbor] = current_node

        return None

    def _is_valid_coordinate(self, coord: Tuple[int, int]) -> bool:
        """
        Check if a coordinate is within the grid bounds.

        Args:
            coord: Tuple of (row, col) coordinates to check.

        Returns:
            True if the coordinate is valid, False otherwise.
        """
        row, col = coord
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighboring nodes (up, down, left, right) that are not walls.

        Args:
            node: Tuple of (row, col) coordinates for the current node.

        Returns:
            List of valid neighboring nodes.
        """
        row, col = node
        neighbors = [
            (row + 1, col),
            (row - 1, col),
            (row, col + 1),
            (row, col - 1)
        ]

        valid_neighbors = []
        for neighbor in neighbors:
            if self._is_valid_coordinate(neighbor) and self.grid[neighbor[0]][neighbor[1]] != 0:
                valid_neighbors.append(neighbor)

        return valid_neighbors

    def _manhattan_distance(self, node1: Tuple[int, int], node2: Tuple[int, int]) -> int:
        """
        Calculate the Manhattan distance between two nodes.

        Args:
            node1: Tuple of (row, col) coordinates for the first node.
            node2: Tuple of (row, col) coordinates for the second node.

        Returns:
            Manhattan distance between node1 and node2.
        """
        return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])
```

## Pytest Tests

Here are 6 pytest tests covering the specified requirements:

```python
import pytest
from typing import List, Tuple, Optional
from astar_grid import AStarGrid

def test_simple_path_uniform_grid():
    """Test finding a path on a uniform grid where all cells have the same cost."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)

    path = astar.find_path(start, end)

    assert path is not None
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
    assert len(path) == 5
    assert path[0] == start and path[-1] == end

def test_path_around_obstacles():
    """Test finding a path that must go around obstacles (walls)."""
    grid = [
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)

    path = astar.find_path(start, end)

    assert path is not None
    assert path == [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)]
    assert len(path) == 5
    assert path[0] == start and path[-1] == end

def test_weighted_grid_path_preference():
    """Test that the path prefers lower-cost cells in a weighted grid."""
    grid = [
        [1, 1, 1],
        [1, 5, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)

    path = astar.find_path(start, end)

    assert path is not None
    # The optimal path should go around the high-cost cell (1,1)
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert len(path) == 5
    assert path[0] == start and path[-1] == end

    # Calculate total cost of the path
    total_cost = sum(grid[row][col] for row, col in path)
    # The minimal possible cost would be 4 (going through the high-cost cell)
    # But since we prefer lower-cost cells, the actual minimal cost is 5
    assert total_cost == 5

def test_no_path_exists_fully_blocked():
    """Test that returns None when no path exists (fully blocked grid)."""
    grid = [
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)

    path = astar.find_path(start, end)

    assert path is None

def test_start_equals_end():
    """Test that returns [start] when start and end are the same."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (1, 1)
    end = (1, 1)

    path = astar.find_path(start, end)

    assert path == [(1, 1)]

def test_invalid_coordinates():
    """Test that raises ValueError for out-of-bounds coordinates."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)

    # Test start out of bounds
    with pytest.raises(ValueError):
        astar.find_path((3, 3), (1, 1))

    # Test end out of bounds
    with pytest.raises(ValueError):
        astar.find_path((1, 1), (3, 3))

    # Test both out of bounds
    with pytest.raises(ValueError):
        astar.find_path((3, 3), (4, 4))

    # Test negative coordinates
    with pytest.raises(ValueError):
        astar.find_path((-1, -1), (1, 1))
```

## Explanation

1. **AStarGrid Class**:
   - Initializes with a grid and validates its structure
   - Implements A* pathfinding algorithm with 4-directional movement

2. **find_path Method**:
   - Returns the optimal path from start to end using A*
   - Handles edge cases:
     - If start == end, returns [start]
     - If start or end is a wall (0), returns None
     - If coordinates are out of bounds, raises ValueError
   - Uses a min-heap (heapq) to manage the open set
   - Returns None if no path exists

3. **Helper Methods**:
   - `_is_valid_coordinate`: Checks if coordinates are within grid bounds
   - `_get_neighbors`: Returns valid adjacent cells (up, down, left, right) that aren't walls
   - `_manhattan_distance`: Calculates the heuristic distance between two nodes

4. **Pytest Tests**:
   - `test_simple_path_uniform_grid`: Verifies basic pathfinding on a uniform grid
   - `test_path_around_obstacles`: Tests pathfinding around walls
   - `test_weighted_grid_path_preference`: Ensures the algorithm prefers lower-cost paths
   - `test_no_path_exists_fully_blocked`: Confirms None is returned when no path exists
   - `test_start_equals_end`: Verifies the trivial case where start and end are the same
   - `test_invalid_coordinates`: Checks that ValueError is raised for invalid coordinates

The implementation uses the standard A* algorithm with:
- g_score: actual cost from start to current node
- f_score: g_score + heuristic (Manhattan distance)
- Priority queue (min-heap) to always expand the most promising node first
- Path reconstruction by tracking came_from relationships

The tests verify both path validity (correct sequence of coordinates) and optimality (minimum total cost).
