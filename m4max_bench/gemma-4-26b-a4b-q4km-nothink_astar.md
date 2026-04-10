# Gemma 4 26B-A4B Q4_K_M — astar

## Output

```python
import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the grid.

        :param grid: A 2D list of integers where 0 is impassable and >0 is the cost to enter.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, r: int, c: int) -> bool:
        """Checks if a cell is within bounds and not a wall."""
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] > 0

    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculates the Manhattan distance heuristic."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from start to end using the A* algorithm.

        :param start: (row, col) starting coordinates.
        :param end: (row, col) target coordinates.
        :return: List of (row, col) tuples representing the path, or None if no path exists.
        :raises ValueError: If start or end coordinates are out of bounds or on a wall.
        """
        # Validate bounds and walls for start/end
        for point in [start, end]:
            r, c = point
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(f"Coordinate {point} is out of bounds.")
            if self.grid[r][c] == 0:
                raise ValueError(f"Coordinate {point} is a wall.")

        if start == end:
            return [start]

        # Priority Queue stores: (priority, current_cost, current_node)
        # priority = current_cost + heuristic
        open_set = []
        heapq.heappush(open_set, (0, 0, start))

        # Track the best cost to reach a node
        g_score = {start: 0}
        # Track the path
        came_from = {}

        while open_set:
            _, current_cost, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # If we found a better path to this node already, skip
            if current_cost > g_score.get(current, float('inf')):
                continue

            r, c = current
            # 4-directional movement: Up, Down, Left, Right
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (r + dr, c + dc)

                if self._is_valid(neighbor[0], neighbor[1]):
                    # Cost to enter the neighbor cell
                    move_cost = self.grid[neighbor[0]][neighbor[1]]
                    tentative_g_score = current_cost + move_cost

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._manhattan_distance(neighbor, end)
                        heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

        return None

# --- Pytest Suite ---
import pytest

def test_path_exists():
    """Test a simple path in a grid with uniform costs."""
    grid = [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 0))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]

def test_weighted_path():
    """Test that the algorithm chooses the lower cost path even if it's longer."""
    grid = [
        [1, 1, 1],
        [1, 9, 1],
        [1, 1, 1]
    ]
    # Path through (1,1) is cost 9. Path around is cost 1+1+1+1+1.
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    # Should avoid the 9
    assert (1, 1) not in path

def test_no_path():
    """Test behavior when the destination is blocked by walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None

def test_start_equals_end():
    """Test behavior when start and end are the same."""
    grid = [[1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_out_of_bounds_error():
    """Test that out of bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_start_error():
    """Test that starting on a wall raises ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="is a wall"):
        astar.find_path((0, 0), (1, 1))

if __name__ == "__main__":
    # This allows running the file directly to see tests pass
    pytest.main([__file__])
```
