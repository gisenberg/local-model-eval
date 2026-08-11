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

    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid, non-wall neighbors in 4 directions."""
        row, col = pos
        neighbors = []
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:  # right, left, down, up
            new_pos = (row + dr, col + dc)
            if self._is_valid(new_pos) and not self._is_wall(new_pos):
                neighbors.append(new_pos)
        return neighbors

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

        # Check walls
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
        start_h = self._manhattan(start, end)
        start_f = start_g + start_h
        heapq.heappush(open_set, (start_f, start_g, start, [start]))

        # Track visited nodes with their best g_score
        visited = {start: start_g}

        while open_set:
            # Pop node with lowest f_score
            f, g, current, path = heapq.heappop(open_set)

            # Check if we reached the end
            if current == end:
                return path

            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                new_g = g + self.grid[neighbor[0]][neighbor[1]]

                # Skip if we've visited with better g_score
                if neighbor in visited and visited[neighbor] <= new_g:
                    continue

                # Calculate f_score
                h = self._manhattan(neighbor, end)
                new_f = new_g + h

                # Update visited and push to open set
                visited[neighbor] = new_g
                new_path = path + [neighbor]
                heapq.heappush(open_set, (new_f, new_g, neighbor, new_path))

        # No path found
        return None


# Test functions
def test_simple_path():
    """Test path on uniform grid."""
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
    # Optimal cost: 5 steps * 1 cost = 5
    total_cost = sum(grid[r][c] for r, c in path)
    assert total_cost == 5
    # Verify path validity (each step is adjacent)
    for i in range(len(path) - 1):
        assert abs(path[i][0] - path[i+1][0]) + abs(path[i][1] - path[i+1][1]) == 1


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
    # Must avoid walls (cost 0)
    for r, c in path:
        assert grid[r][c] != 0
    # Optimal cost: 5 steps * 1 cost = 5
    total_cost = sum(grid[r][c] for r, c in path)
    assert total_cost == 5


def test_weighted_grid():
    """Test weighted grid prefers lower-cost cells."""
    grid = [
        [1, 1, 1, 1],
        [1, 5, 5, 1],
        [1, 1, 1, 1]
    ]
    path = AStarGrid(grid).find_path((0, 0), (2, 3))

    assert path is not None
    # Optimal path should avoid the expensive middle row
    # Path: (0,0) -> (0,1) -> (0,2) -> (0,3) -> (1,3) -> (2,3) = 6 cost
    # Alternative through middle: (0,0) -> (1,0) -> (1,1) -> (1,2) -> (1,3) -> (2,3) = 1+5+5+5+1+1 = 18
    total_cost = sum(grid[r][c] for r, c in path)
    assert total_cost == 6
    # Verify path doesn't go through expensive cells
    assert (1, 1) not in path
    assert (1, 2) not in path


def test_no_path():
    """Test fully blocked grid returns None."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    # Blocked with wall in middle
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert path is None

    # Fully blocked
    grid2 = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    path2 = AStarGrid(grid2).find_path((0, 0), (2, 2))
    assert path2 is None


def test_start_equals_end():
    """Test when start equals end."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    path = AStarGrid(grid).find_path((1, 1), (1, 1))
    assert path == [(1, 1)]


def test_invalid_coordinates():
    """Test out of bounds raises ValueError."""
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

    # Wall start returns None
    grid_wall = [
        [0, 1],
        [1, 1]
    ]
    astar_wall = AStarGrid(grid_wall)
    assert astar_wall.find_path((0, 0), (1, 1)) is None

    # Wall end returns None
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
2. **`find_path`** method implementing A* with heapq
3. **4-directional movement** (up, down, left, right)
4. **Manhattan distance heuristic**
5. **Edge cases handled**: start==end, walls, out of bounds
6. **Optimal path** guaranteed by A* algorithm
7. **Type hints** throughout
8. **Docstrings** on all methods
9. **6 pytest tests** covering all required scenarios

The algorithm uses:
- `g_score`: actual cost from start to current node
- `h_score`: Manhattan distance heuristic to end
- `f_score`: g + h (total estimated cost)
- Min-heap to always expand the node with lowest f_score
- Visited dictionary to track best g_score for each node (prevents revisiting with worse cost)

The path is guaranteed optimal because A* with an admissible heuristic (Manhattan distance is admissible for 4-directional movement) always finds the shortest path.