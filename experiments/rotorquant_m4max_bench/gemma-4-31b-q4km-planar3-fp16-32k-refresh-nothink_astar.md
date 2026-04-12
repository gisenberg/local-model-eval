# Gemma 4 31B-IT Q4_K_M (planar3/f16 @ 32K, new base default ub) — astar

**Hypothesis:** Refresh clean comparison: is rotorquant K-only still faster than f16 at default ub on Gemma 31B?

## Output

```python
import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    Implements A* pathfinding on a weighted 2D grid.
    Grid values: 0 = wall, positive integer = cost to enter the cell.
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

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from start to end using A*.
        :param start: (row, col) starting coordinates.
        :param end: (row, col) ending coordinates.
        :return: List of coordinates representing the path, or None if no path exists.
        :raises ValueError: If start or end are out of bounds.
        """
        # Bounds checking
        for pt in (start, end):
            if not (0 <= pt[0] < self.rows and 0 <= pt[1] < self.cols):
                raise ValueError(f"Point {pt} is out of grid bounds.")

        # Start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        if start == end:
            return [start]

        # Priority Queue: (priority, current_cost, current_node)
        # priority = current_cost + heuristic
        open_set = []
        heapq.heappush(open_set, (0 + self._heuristic(start, end), 0, start))
        
        # Track the cheapest cost to reach a node
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        # Track the path
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
                    tentative_g_score = current_cost + self.grid[neighbor[0]][neighbor[1]]

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        g_score[neighbor] = tentative_g_score
                        priority = tentative_g_score + self._heuristic(neighbor, end)
                        came_from[neighbor] = current
                        heapq.heappush(open_set, (priority, tentative_g_score, neighbor))

        return None

# --- Pytest Tests ---
import pytest

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_simple_path():
    # 1s are cheap, 0s are walls
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

def test_no_path():
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
        astar.find_path((-1, 0), (0, 0))
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
    assert astar.find_path((1, 1), (0, 0)) is None
```

### Implementation Details:
1.  **Complexity**: The time complexity is $O(E \log V)$ where $E$ is the number of edges (roughly $4 \times \text{cells}$) and $V$ is the number of cells.
2.  **Optimality**: By using the Manhattan distance (which is an *admissible* heuristic for 4-directional movement) and a priority queue, A* is guaranteed to find the path with the minimum total cost.
3.  **Weighted Logic**: The cost is calculated as the sum of the values of the cells entered. The starting cell's value is not added to the total cost (as you are already there), which is standard for grid pathfinding.
4.  **Edge Cases**:
    *   **Start == End**: Returns a list containing only the start node.
    *   **Walls**: Cells with `0` are treated as impassable.
    *   **Bounds**: `ValueError` is raised if coordinates are outside the grid dimensions.
    *   **No Path**: Returns `None` if the priority queue empties without reaching the destination.
