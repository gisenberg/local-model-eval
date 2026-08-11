Here's a complete implementation of A* pathfinding on a weighted 2D grid with all the requested features:

```python
from typing import List, Tuple, Optional
import heapq


class AStarGrid:
    """
    A* pathfinding algorithm for weighted 2D grids.

    Grid values represent movement cost (0 = impassable wall, positive int = cost to enter that cell).
    Supports 4-directional movement (up, down, left, right) with Manhattan distance heuristic.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid.

        Args:
            grid: 2D list where grid[row][col] = movement cost (0 = wall, positive = cost)
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a wall (cost = 0)."""
        row, col = pos
        return self.grid[row][col] == 0

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions (4-directional movement)."""
        row, col = pos
        neighbors = []
        # Up, down, left, right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_pos = (row + dr, col + dc)
            if self._is_valid(new_pos) and not self._is_wall(new_pos):
                neighbors.append(new_pos)
        return neighbors

    def _manhattan_distance(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.

        Args:
            start: Starting position (row, col)
            end: Target position (row, col)

        Returns:
            List of coordinates from start to end inclusive, or None if no path exists.

        Raises:
            ValueError: If start or end are out of bounds
        """
        # Validate bounds
        if not self._is_valid(start):
            raise ValueError(f"Start position {start} is out of bounds")
        if not self._is_valid(end):
            raise ValueError(f"End position {end} is out of bounds")

        # Check if start or end are walls
        if self._is_wall(start):
            return None
        if self._is_wall(end):
            return None

        # Start equals end
        if start == end:
            return [start]

        # Initialize open set with start node
        # Format: (f_score, g_score, position, path)
        open_set = []
        start_g = self.grid[start[0]][start[1]]
        start_h = self._manhattan_distance(start, end)
        start_f = start_g + start_h
        heapq.heappush(open_set, (start_f, start_g, start, [start]))

        # Track visited nodes with their best g_score
        visited = {start: start_g}

        while open_set:
            # Pop node with lowest f_score
            f_score, g_score, current, path = heapq.heappop(open_set)

            # Check if we reached the end
            if current == end:
                return path

            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                # Calculate new g_score for neighbor
                new_g = g_score + self.grid[neighbor[0]][neighbor[1]]

                # Check if we've visited this neighbor with a better score
                if neighbor in visited and visited[neighbor] <= new_g:
                    continue

                # Update visited with better score
                visited[neighbor] = new_g

                # Calculate f_score
                h_score = self._manhattan_distance(neighbor, end)
                new_f = new_g + h_score

                # Add to open set with path
                new_path = path + [neighbor]
                heapq.heappush(open_set, (new_f, new_g, neighbor, new_path))

        # No path found
        return None


# Test functions
def test_simple_path():
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

    # Calculate total cost
    total_cost = sum(grid[r][c] for r, c in path)
    assert total_cost == 5  # 3 steps each costing 1

    # Verify path validity (each step is adjacent)
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1


def test_path_around_obstacles():
    """Test path around obstacles."""
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
    for r, c in path:
        assert grid[r][c] != 0

    # Verify path validity
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1


def test_weighted_grid():
    """Test weighted grid where path prefers lower-cost cells."""
    grid = [
        [1, 1, 1, 1],
        [1, 5, 5, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 3))

    assert path is not None

    # Calculate total cost
    total_cost = sum(grid[r][c] for r, c in path)

    # Optimal path should avoid the expensive middle row
    # Expected path: (0,0) -> (0,1) -> (0,2) -> (0,3) -> (1,3) -> (2,3)
    # Cost: 1+1+1+1+1+1 = 6
    assert total_cost == 6

    # Verify path validity
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1


def test_no_path():
    """Test when no path exists (fully blocked)."""
    grid = [
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1]
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
    try:
        astar.find_path((5, 5), (0, 0))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Out of bounds end
    try:
        astar.find_path((0, 0), (5, 5))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Wall start
    grid_wall = [
        [0, 1],
        [1, 1]
    ]
    astar_wall = AStarGrid(grid_wall)
    assert astar_wall.find_path((0, 0), (1, 1)) is None

    # Wall end
    assert astar_wall.find_path((1, 1), (0, 0)) is None


# Run tests if executed directly
if __name__ == "__main__":
    test_simple_path()
    test_path_around_obstacles()
    test_weighted_grid()
    test_no_path()
    test_start_equals_end()
    test_invalid_coordinates()
    print("All tests passed!")
```

This implementation includes:

1. **Class `AStarGrid`** with proper initialization
2. **`find_path` method** returning optimal path or None
3. **4-directional movement** (up, down, left, right)
4. **Manhattan distance heuristic**
5. **Edge case handling**:
   - Start == end returns `[start]`
   - Start or end walls return None
   - Out of bounds raises ValueError
6. **Optimal path** guaranteed by A* algorithm with admissible heuristic
7. **Min-heap** using `heapq` for open set
8. **Type hints** throughout and docstrings on each method
9. **6 pytest tests** covering all required scenarios with path validity and optimality checks

The algorithm maintains optimality because:
- The Manhattan distance heuristic is admissible (never overestimates the true cost)
- We use a min-heap to always explore the most promising node first
- We track visited nodes with their best g-score to avoid revisiting with worse scores

The path returned is guaranteed to be optimal (minimum total cost) because A* with an admissible heuristic guarantees optimality.