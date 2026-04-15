# Ministral-3-14B-Instruct-2512-BF16 (L40S vLLM) — A* Pathfinding — Run 2

## Output

Here's the implementation of the A* pathfinding algorithm for a weighted 2D grid according to your requirements:

```python
import heapq
from typing import List, Tuple, Optional, Dict, Set
from collections import deque

class AStarGrid:
    """A weighted 2D grid for A* pathfinding with movement costs."""

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid with movement costs.

        Args:
            grid: 2D list where each value represents movement cost (0 = impassable wall)
        """
        self.grid = grid
        self.rows = len(grid)
        if self.rows == 0:
            raise ValueError("Grid cannot be empty")
        self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start: Starting coordinates (row, col)
            end: Target coordinates (row, col)

        Returns:
            List of coordinates representing the path, or None if no path exists

        Raises:
            ValueError: If start or end coordinates are out of bounds
        """
        # Validate coordinates
        if not self._is_valid_coordinate(start) or not self._is_valid_coordinate(end):
            raise ValueError("Start or end coordinates are out of bounds")

        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # If start equals end
        if start == end:
            return [start]

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Priority queue: (f_score, g_score, current_node)
        open_set: List[Tuple[float, float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))

        # For node n, came_from[n] is the node immediately preceding it on the cheapest path
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # For node n, g_score[n] is the cost of the cheapest path from start to n
        g_score: Dict[Tuple[int, int], float] = {start: 0}

        # For node n, f_score[n] = g_score[n] + h(n)
        f_score: Dict[Tuple[int, int], float] = {start: self._manhattan_distance(start, end)}

        open_set_hash: Set[Tuple[int, int]] = {start}

        while open_set:
            current = heapq.heappop(open_set)[2]
            open_set_hash.remove(current)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._is_valid_coordinate(neighbor):
                    continue
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue

                # The distance from start to a neighbor is the distance from start to the current
                # node plus the weight of the edge from current to neighbor
                tentative_g_score = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._manhattan_distance(neighbor, end)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        return None  # No path found

    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_valid_coordinate(self, coord: Tuple[int, int]) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= coord[0] < self.rows and 0 <= coord[1] < self.cols

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]],
                         current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from came_from dictionary."""
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return path[::-1]  # Reverse to get start to end
```

And here are the pytest tests covering all the requirements:

```python
import pytest
from typing import List, Tuple, Optional

def test_simple_path_uniform_grid():
    """Test simple path on uniform grid (all cells have same cost)."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    a_star = AStarGrid(grid)
    path = a_star.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert sum(grid[r][c] for r, c in path) == 5  # Optimal cost

def test_path_around_obstacles():
    """Test path that goes around obstacles."""
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 1, 1],
        [1, 1, 1, 1]
    ]
    a_star = AStarGrid(grid)
    path = a_star.find_path((0, 0), (3, 3))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (3, 2), (3, 3)]
    assert sum(grid[r][c] for r, c in path) == 6  # Optimal cost

def test_weighted_grid():
    """Test path prefers lower-cost cells."""
    grid = [
        [1, 2, 1],
        [1, 1, 1],
        [1, 2, 1]
    ]
    a_star = AStarGrid(grid)
    path = a_star.find_path((0, 0), (2, 2))
    # Should prefer path through center (cost 3) over path through corners (cost 4)
    assert path == [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)]
    assert sum(grid[r][c] for r, c in path) == 3

def test_no_path_exists():
    """Test when no path exists (fully blocked)."""
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    a_star = AStarGrid(grid)
    assert a_star.find_path((0, 0), (2, 2)) is None

def test_start_equals_end():
    """Test when start equals end."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    a_star = AStarGrid(grid)
    assert a_star.find_path((1, 1), (1, 1)) == [(1, 1)]

def test_invalid_coordinates():
    """Test invalid coordinates raise ValueError."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    a_star = AStarGrid(grid)
    with pytest.raises(ValueError):
        a_star.find_path((0, 0), (2, 2))
    with pytest.raises(ValueError):
        a_star.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        a_star.find_path((1, 1), (1, 2))
```

The implementation includes:
1. A proper A* algorithm with Manhattan distance heuristic
2. 4-directional movement (no diagonals)
3. Handling of all edge cases mentioned
4. Type hints throughout
5. Min-heap implementation using heapq
6. Optimal path finding (minimum total cost)
7. Proper docstrings for all methods

The tests cover:
- Simple path on uniform grid
- Path around obstacles
- Weighted grid (path prefers lower-cost cells)
- No path exists case
- Start equals end case
- Invalid coordinates case

The implementation ensures that the path is always optimal (minimum total cost) when a path exists.
