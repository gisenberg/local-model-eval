# Gemma 4 26B-A4B Q6_K (turbo4 KV) — astar

## Output

```python
import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    Implements the A* pathfinding algorithm on a weighted 2D grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the grid.
        
        :param grid: A 2D list where 0 represents an impassable wall and 
                     positive integers represent the cost to enter that cell.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from start to end using A*.

        :param start: (row, col) starting coordinates.
        :param end: (row, col) target coordinates.
        :return: A list of (row, col) tuples representing the path, or None if no path exists.
        :raises ValueError: If start or end are out of bounds or are walls.
        """
        # Bounds and Wall Validation
        for point in [start, end]:
            r, c = point
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(f"Point {point} is out of bounds.")
            if self.grid[r][c] == 0:
                raise ValueError(f"Point {point} is an impassable wall.")

        if start == end:
            return [start]

        # Priority Queue stores: (f_score, current_node)
        # f_score = g_score (cost from start) + h_score (estimated cost to end)
        open_set = []
        heapq.heappush(open_set, (0, start))

        # Tracking costs and paths
        came_from = {}
        g_score = {start: 0}
        
        # Set for faster lookup of nodes in open_set
        open_set_hash = {start}

        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)

            if current == end:
                return self._reconstruct_path(came_from, current)

            r, c = current
            # 4-directional movement (Up, Down, Left, Right)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (r + dr, c + dc)
                nr, nc = neighbor

                # Check bounds and if it's a wall
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    weight = self.grid[nr][nc]
                    if weight == 0:
                        continue

                    tentative_g_score = g_score[current] + weight

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._heuristic(neighbor, end)
                        
                        if neighbor not in open_set_hash:
                            heapq.heappush(open_set, (f_score, neighbor))
                            open_set_hash.add(neighbor)

        return None

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstructs the path from the came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


# --- Pytest Tests ---
import pytest

def test_simple_path():
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

def test_weighted_path():
    # Path through (0,1) is cost 10, path through (1,0) is cost 1.
    # We want the path with minimum total cost.
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    # Should go around the expensive wall: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2)
    # or similar, but definitely not through (0,1)
    assert (0, 1) not in path

def test_no_path():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None

def test_start_is_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_out_of_bounds():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_start_is_wall():
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))

if __name__ == "__main__":
    # Manual execution if not using pytest command
    pytest.main([__file__])
```
