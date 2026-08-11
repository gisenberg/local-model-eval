Here's a complete implementation of A* pathfinding on a weighted 2D grid with all the required features:

```python
from typing import List, Tuple, Optional, Dict
import heapq


class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.

    Grid values represent movement cost (0 = impassable wall, positive int = cost to enter that cell).
    Supports 4-directional movement (up, down, left, right) - no diagonals.
    Uses Manhattan distance as heuristic.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid.

        Args:
            grid: 2D list where grid[row][col] is the movement cost.
                  0 = impassable wall, positive int = cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within grid bounds."""
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is a wall (impassable)."""
        row, col = pos
        return self.grid[row][col] == 0

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighbors for 4-directional movement."""
        row, col = pos
        neighbors = [
            (row - 1, col),  # up
            (row + 1, col),  # down
            (row, col - 1),  # left
            (row, col + 1)   # right
        ]
        return [n for n in neighbors if self._is_valid(n) and not self._is_wall(n)]

    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.

        Args:
            start: (row, col) coordinates of start position
            end: (row, col) coordinates of end position

        Returns:
            List of (row, col) coordinates from start to end inclusive, or None if no path exists.

        Raises:
            ValueError: If start or end is out of bounds
        """
        # Validate inputs
        if not self._is_valid(start):
            raise ValueError(f"Start position {start} is out of bounds")
        if not self._is_valid(end):
            raise ValueError(f"End position {end} is out of bounds")

        # Check if start or end is a wall
        if self._is_wall(start):
            return None
        if self._is_wall(end):
            return None

        # Handle start == end case
        if start == end:
            return [start]

        # A* algorithm
        open_set = []  # min-heap: (f_score, g_score, position)
        heapq.heappush(open_set, (0, 0, start))

        # Track g scores (cost from start to node)
        g_scores: Dict[Tuple[int, int], int] = {start: 0}

        # Track path reconstruction
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_set:
            current_f, current_g, current = heapq.heappop(open_set)

            # Skip if we've found a better path to this node
            if current_g > g_scores.get(current, float('inf')):
                continue

            # Check if we reached the end
            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                # Calculate tentative g score for neighbor
                tentative_g = current_g + self.grid[neighbor[0]][neighbor[1]]

                # If this path is better than previous, update
                if tentative_g < g_scores.get(neighbor, float('inf')):
                    g_scores[neighbor] = tentative_g
                    came_from[neighbor] = current

                    # Calculate f score = g + heuristic
                    h = self._manhattan_distance(neighbor, end)
                    f = tentative_g + h

                    heapq.heappush(open_set, (f, tentative_g, neighbor))

        # No path found
        return None

    def get_path_cost(self, path: List[Tuple[int, int]]) -> int:
        """
        Calculate the total cost of a path (excluding start cell).

        Args:
            path: List of (row, col) coordinates

        Returns:
            Total movement cost of the path
        """
        if not path:
            return 0
        # Sum costs of all cells except the start (start cost is not counted)
        return sum(self.grid[row][col] for row, col in path[1:])


# Test cases
import pytest

def test_simple_path_uniform_grid():
    """Test simple path on uniform grid."""
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
    assert astar.get_path_cost(path) == 4  # 4 steps through cost-1 cells

    # Verify path validity
    for i in range(len(path) - 1):
        row_diff = abs(path[i+1][0] - path[i][0])
        col_diff = abs(path[i+1][1] - path[i][1])
        assert row_diff + col_diff == 1  # 4-directional movement

def test_path_around_obstacles():
    """Test path around obstacles."""
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (3, 3))

    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (3, 3)

    # Path should go around the obstacle
    for pos in path:
        row, col = pos
        assert grid[row][col] != 0  # No walls in path

    # Optimal path cost: 6 steps through cost-1 cells
    assert astar.get_path_cost(path) == 6

def test_weighted_grid_prefers_lower_cost():
    """Test that path prefers lower-cost cells."""
    grid = [
        [1, 1, 1, 1],
        [1, 5, 5, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (3, 3))

    assert path is not None

    # The optimal path should avoid the high-cost cells (5)
    # It should go around them instead
    for pos in path:
        row, col = pos
        assert grid[row][col] != 5  # Should avoid cost-5 cells

    # Optimal cost: 6 steps through cost-1 cells
    assert astar.get_path_cost(path) == 6

def test_no_path_exists_fully_blocked():
    """Test when no path exists (fully blocked)."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)

    # Blocked path from top-left to bottom-right
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

    # Also test with completely blocked grid
    grid2 = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    astar2 = AStarGrid(grid2)
    path2 = astar2.find_path((0, 0), (2, 2))
    assert path2 is None

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
    assert astar.get_path_cost(path) == 0

def test_invalid_coordinates():
    """Test invalid coordinates raise ValueError."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)

    # Out of bounds start
    with pytest.raises(ValueError):
        astar.find_path((3, 0), (1, 1))

    # Out of bounds end
    with pytest.raises(ValueError):
        astar.find_path((1, 1), (0, 3))

    # Both out of bounds
    with pytest.raises(ValueError):
        astar.find_path((5, 5), (6, 6))

def test_start_or_end_is_wall():
    """Test when start or end is a wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)

    # Start is a wall
    path = astar.find_path((1, 1), (0, 0))
    assert path is None

    # End is a wall
    path = astar.find_path((0, 0), (1, 1))
    assert path is None

    # Both are walls
    path = astar.find_path((1, 1), (1, 1))
    assert path is None

def test_optimality_with_weights():
    """Test that path is truly optimal with varying weights."""
    grid = [
        [1, 1, 1, 1],
        [1, 2, 2, 1],
        [1, 2, 2, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)

    # Test multiple paths and verify optimality
    path = astar.find_path((0, 0), (3, 3))
    assert path is not None

    # Calculate cost and verify it's the minimum possible
    cost = astar.get_path_cost(path)

    # The optimal path should go around the 2-cost cells
    # Path: (0,0) -> (0,1) -> (0,2) -> (0,3) -> (1,3) -> (2,3) -> (3,3)
    # Cost: 1 + 1 + 1 + 1 + 1 + 1 = 6
    assert cost == 6

    # Verify path doesn't go through cost-2 cells
    for pos in path:
        row, col = pos
        assert grid[row][col] != 2
```

This implementation includes:

1. **Class `AStarGrid`** with proper initialization
2. **`find_path` method** that returns the optimal path or None
3. **4-directional movement** support
4. **Manhattan distance heuristic**
5. **Edge case handling** (start == end, walls, out of bounds)
6. **Min-heap** for open set using `heapq`
7. **Type hints** throughout
8. **Docstrings** on each method
9. **7 pytest tests** covering all required scenarios

The tests verify:
- Simple path on uniform grid
- Path around obstacles
- Weighted grid path preference
- No path exists (fully blocked)
- Start equals end
- Invalid coordinates (raises ValueError)
- Start or end is a wall
- Optimality with varying weights

The algorithm guarantees optimality because:
- It uses A* with admissible heuristic (Manhattan distance)
- The heuristic never overestimates the actual cost
- It explores nodes in order of f-score (g + h)
- It maintains proper g-scores and updates them when better paths are found