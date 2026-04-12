# Gemma 4 26B-A4B Q6_K (planar3/f16 @ 32K, new base default ub) — astar

**Hypothesis:** Refresh recommended: rotorquant K-only at 32K at default ub

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
        
        :param grid: A 2D list where 0 represents an impassable wall 
                     and positive integers represent the cost to enter that cell.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Manhattan distance heuristic.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from start to end using the A* algorithm.

        :param start: (row, col) starting coordinates.
        :param end: (row, col) target coordinates.
        :return: A list of (row, col) tuples representing the path, or None if no path exists.
        :raises ValueError: If start or end coordinates are out of bounds or on a wall.
        """
        # Bounds and Wall Validation
        for point in [start, end]:
            r, c = point
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(f"Coordinate {point} is out of bounds.")
            if self.grid[r][c] == 0:
                raise ValueError(f"Coordinate {point} is an impassable wall.")

        if start == end:
            return [start]

        # Priority Queue stores: (priority, current_cost, current_node)
        # priority = current_cost + heuristic
        open_set = []
        heapq.heappush(open_set, (0, 0, start))

        # Trackers
        came_from = {}
        g_score = {start: 0}
        
        # Directions: Up, Down, Left, Right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            _, current_cost, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            # If we found a better path to this node already, skip
            if current_cost > g_score.get(current, float('inf')):
                continue

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                r, c = neighbor

                # Check bounds and if it's a wall
                if 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] > 0:
                    # Cost to enter the neighbor cell
                    tentative_g_score = current_cost + self.grid[r][c]

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

        return None

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstructs the path from the came_from dictionary.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


# --- Pytest Tests ---
import pytest

def test_path_exists():
    grid = [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 0))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]

def test_no_path():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None

def test_start_is_end():
    grid = [[5]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_weighted_optimal_path():
    # Path A: (0,0)->(0,1)->(0,2) cost: 1 + 10 + 1 = 12
    # Path B: (0,0)->(1,0)->(1,1)->(1,2)->(0,2) cost: 1 + 1 + 1 + 1 + 1 = 5
    grid = [
        [1, 10, 1],
        [1,  1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    expected = [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]
    assert path == expected

def test_out_of_bounds_raises_error():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_start_raises_error():
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="impassable wall"):
        astar.find_path((0, 0), (1, 1))

if __name__ == "__main__":
    # To run tests manually without pytest command
    pytest.main([__file__])
```
