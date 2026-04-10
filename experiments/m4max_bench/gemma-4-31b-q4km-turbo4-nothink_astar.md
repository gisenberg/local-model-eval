# Gemma 4 31B-IT Q4_K_M (turbo4 KV) — astar

## Output

```python
import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    Implements A* pathfinding on a weighted 2D grid.
    Grid values: 0 = wall, >0 = cost to enter the cell.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.
        :param grid: A 2D list of integers representing movement costs.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _get_heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from start to end using A*.
        :param start: (row, col) starting coordinates.
        :param end: (row, col) ending coordinates.
        :return: List of coordinates from start to end, or None if no path exists.
        :raises ValueError: If start or end are out of bounds.
        """
        # Bounds checking
        for point in (start, end):
            r, c = point
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(f"Point {point} is out of grid bounds.")

        if start == end:
            return [start]

        # If start or end is a wall, no path is possible
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Priority Queue: (priority, current_cost, current_node)
        # priority = current_cost + heuristic
        open_set = []
        heapq.heappush(open_set, (0 + self._get_heuristic(start, end), 0, start))
        
        # Tracking costs and paths
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        while open_set:
            _, current_cost, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            # 4-directional movement
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                # Bounds and Wall check
                if (0 <= neighbor[0] < self.rows and 
                    0 <= neighbor[1] < self.cols and 
                    self.grid[neighbor[0]][neighbor[1]] != 0):
                    
                    # Cost to enter the neighbor cell
                    weight = self.grid[neighbor[0]][neighbor[1]]
                    tentative_g_score = g_score[current] + weight

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._get_heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

        return None

# --- Pytest Tests ---
import pytest

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_simple_path():
    # 1 1 1
    # 0 0 1
    # 1 1 1
    grid = [
        [1, 1, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    expected = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]
    assert astar.find_path((0, 0), (2, 0)) == expected

def test_weighted_path():
    # Path A: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) | Cost: 1+1+1+1 = 4
    # Path B: (0,0)->(1,0)->(2,0)->(2,1)->(2,2) | Cost: 10+10+1+1 = 22
    grid = [
        [1, 1, 1],
        [10, 0, 1],
        [10, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

def test_no_path_walls():
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) is None

def test_out_of_bounds():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))

def test_start_or_end_is_wall():
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    # Start is wall
    assert astar.find_path((0, 0), (1, 1)) is None
    # End is wall
    grid2 = [[1, 0], [1, 1]]
    astar2 = AStarGrid(grid2)
    assert astar2.find_path((0, 0), (0, 1)) is None
```

### Implementation Details:
1.  **Time Complexity**: $O(E \log V)$ where $E$ is the number of edges (roughly $4 \times \text{cells}$) and $V$ is the number of cells.
2.  **Space Complexity**: $O(V)$ to store the `g_score` and `came_from` maps.
3.  **Optimality**: Since the Manhattan distance is an **admissible heuristic** (it never overestimates the distance to the goal on a grid with costs $\ge 1$), A* is guaranteed to find the optimal path.
4.  **Edge Cases**:
    *   **Start == End**: Returns a list containing only the start node.
    *   **Walls**: Cells with value `0` are treated as impassable.
    *   **Weights**: The algorithm uses the value of the cell being entered as the edge weight, ensuring the path with the minimum total cost is chosen.
    *   **Bounds**: `ValueError` is raised if coordinates are outside the grid dimensions.
