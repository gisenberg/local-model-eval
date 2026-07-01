```python
from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.

    Grid values:
        - 0 : impassable wall
        - positive int : cost to enter that cell

    Movement: 4-directional (up, down, left, right), no diagonals.
    Heuristic: Manhattan distance.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid.

        Args:
            grid: 2D list of integers representing movement costs.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_in_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within the grid boundaries."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is a wall (cost == 0)."""
        r, c = pos
        return self.grid[r][c] == 0

    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Compute Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A*.

        Args:
            start: (row, col) of the starting cell.
            end:   (row, col) of the destination cell.

        Returns:
            List of (row, col) coordinates from start to end inclusive,
            or None if no path exists.

        Raises:
            ValueError: if start or end are out of bounds.
        """
        # Validate start and end positions
        if not self._is_in_bounds(start) or not self._is_in_bounds(end):
            raise ValueError("Start or end position is out of bounds")

        # Check if start or end are walls
        if self._is_wall(start) or self._is_wall(end):
            return None

        # Trivial case: start == end
        if start == end:
            return [start]

        # A* algorithm
        open_set = []  # min-heap of (f, g, (r, c))
        heapq.heappush(open_set, (0, 0, start))

        # g_cost: best known cost from start to a cell
        g_cost = {start: 0}
        # came_from: parent mapping for path reconstruction
        came_from = {}

        # Closed set (visited cells)
        closed = set()

        while open_set:
            f, g, current = heapq.heappop(open_set)

            if current in closed:
                continue
            closed.add(current)

            # Goal reached
            if current == end:
                # Reconstruct path
                path = []
                node = end
                while node != start:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                path.reverse()
                return path

            # Explore neighbors (4-directional)
            r, c = current
            neighbors = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]

            for nb in neighbors:
                if not self._is_in_bounds(nb):
                    continue
                if self._is_wall(nb):
                    continue

                # Tentative g cost
                tentative_g = g + self.grid[nb[0]][nb[1]]

                # If this path is better than previously known
                if nb not in g_cost or tentative_g < g_cost[nb]:
                    g_cost[nb] = tentative_g
                    h = self._manhattan_distance(nb, end)
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, tentative_g, nb))
                    came_from[nb] = current

        # No path found
        return None


# =============================================================================
# Pytest tests
# =============================================================================

import pytest

def test_simple_path_uniform_grid():
    """Simple path on a uniform grid (all 1s)."""
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
    # Path must be contiguous and valid
    for i in range(len(path)-1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        assert abs(r1-r2) + abs(c1-c2) == 1  # 4-directional step
        assert grid[r2][c2] > 0  # not a wall
    assert path[0] == start
    assert path[-1] == end
    # Optimal cost: Manhattan distance * 1 = 4 steps, cost = 4 (since all cells cost 1)
    # But path length (number of cells) = 5, cost = sum of costs of all cells except start? Actually cost to enter a cell, so total cost = sum of costs of all cells except start? Usually A* counts cost of entering a cell, so start cell cost is not counted. But we defined g as sum of costs of entered cells. So total cost = sum of grid values of all cells except start. For uniform 1, path length = 5 cells, cost = 4. Let's compute.
    total_cost = sum(grid[r][c] for (r,c) in path[1:])  # exclude start
    assert total_cost == 4  # 4 steps * 1

def test_path_around_obstacles():
    """Path must go around a wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (0, 2)
    path = astar.find_path(start, end)
    assert path is not None
    # Must go around the wall at (1,1)
    # Possible paths: (0,0)->(0,1)->(0,2) is blocked? No, (0,1) is 1, (0,2) is 1, but (1,1) is wall but not on that path. Actually (0,0)->(0,1)->(0,2) is valid because (0,1) is 1. So that's the shortest. But if we want to test obstacle, we need a wall that forces detour. Let's use a wall that blocks direct path.
    grid2 = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 0, 1]
    ]
    # start (0,0), end (2,2) -> direct path would go through (1,1) and (2,1) but (2,1) is wall. So must go around.
    astar2 = AStarGrid(grid2)
    start2 = (0,0)
    end2 = (2,2)
    path2 = astar2.find_path(start2, end2)
    assert path2 is not None
    # Check no wall cells in path
    for (r,c) in path2:
        assert grid2[r][c] != 0
    assert path2[0] == start2
    assert path2[-1] == end2
    # Optimal cost: Manhattan distance = 4, but actual cost must be >4 because detour. Let's compute.
    total_cost = sum(grid2[r][c] for (r,c) in path2[1:])
    # The optimal path: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) cost = 1+1+1+1 = 4? Actually (0,0) start, then (0,1) cost1, (0,2) cost1, (1,2) cost1, (2,2) cost1 = total 4. But Manhattan distance is 4, so cost equals heuristic? That's fine. But we need a case where detour increases cost. Let's use a more complex grid.
    # Instead, let's test a grid where direct path is blocked by a wall and detour adds extra cost.
    grid3 = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    astar3 = AStarGrid(grid3)
    start3 = (0,0)
    end3 = (0,3)
    path3 = astar3.find_path(start3, end3)
    assert path3 is not None
    # Direct path along top row is blocked? (0,1) is 1, (0,2) is 1, (0,3) is 1, so it's fine. Actually the walls are at (1,1) and (1,2), not on top row. So direct path exists. Let's make a proper obstacle: start (0,0), end (2,2) with walls at (1,1) and (1,2) forcing detour.
    grid4 = [
        [1, 1, 1],
        [1, 0, 0],
        [1, 1, 1]
    ]
    astar4 = AStarGrid(grid4)
    start4 = (0,0)
    end4 = (2,2)
    path4 = astar4.find_path(start4, end4)
    assert path4 is not None
    # Must go around via (0,1)->(0,2)->(1,2)->(2,2) or (0,1)->(1,1) blocked, so only (0,1)->(0,2)->(1,2)->(2,2). Cost = 1+1+1+1 = 4. Manhattan distance = 4, still same. To force higher cost, use non-uniform costs.
    # Let's use a weighted grid test instead.
    # For this test, just ensure path avoids walls.
    for (r,c) in path4:
        assert grid4[r][c] != 0
    assert path4[0] == start4
    assert path4[-1] == end4

def test_weighted_grid_prefers_lower_cost():
    """Path should prefer lower-cost cells even if longer."""
    grid = [
        [1, 100, 100],
        [1, 100, 100],
        [1,   1,   1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (0, 2)
    # Direct path: (0,0)->(0,1)->(0,2) cost = 100+100 = 200
    # Alternative: go down to (1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2) cost = 1+1+1+1+100+100? Wait compute: (0,0) start, then (1,0) cost1, (2,0) cost1, (2,1) cost1, (2,2) cost1, (1,2) cost100, (0,2) cost100 = total 1+1+1+1+100+100 = 204. That's worse. Actually there is a better path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2) is 204. But maybe go (0,0)->(1,0)->(1,1)? (1,1) is 100, so no. So direct is cheaper. Let's design a grid where detour is cheaper.
    grid2 = [
        [1, 100, 100],
        [1, 100, 100],
        [1,   1,   1]
    ]
    # Actually we want a case where going around avoids high cost cells. For example, start (0,0), end (2,2). Direct path through (1,1) costs 100, but going around via bottom row costs 1+1+1+1+1+1? Let's compute: (0,0)->(1,0)->(2,0)->(2,1)->(2,2) cost = 1+1+1+1 = 4 (since (2,2) is 1). That's cheaper than going through (1,1) which would be 1+100+1 = 102. So the algorithm should choose the bottom path.
    grid2 = [
        [1, 100, 100],
        [1, 100, 100],
        [1,   1,   1]
    ]
    astar2 = AStarGrid(grid2)
    start2 = (0,0)
    end2 = (2,2)
    path2 = astar2.find_path(start2, end2)
    assert path2 is not None
    # Path should not include (1,1) because it's high cost. Check that path goes via bottom.
    # Expected path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2) or similar.
    # Verify no cell with cost 100 in path (except maybe start? start is 1)
    for (r,c) in path2:
        assert grid2[r][c] != 100  # should avoid high cost
    total_cost = sum(grid2[r][c] for (r,c) in path2[1:])
    # Optimal cost = 4 (since 4 steps of cost 1)
    assert total_cost == 4

def test_no_path_fully_blocked():
    """No path exists when walls block all routes."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)
    path = astar.find_path(start, end)
    assert path is None

def test_start_equals_end():
    """Path of length 1 when start == end."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (0, 0)
    path = astar.find_path(start, end)
    assert path == [start]

def test_invalid_coordinates():
    """Out of bounds start/end raises ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 5), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 0))
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))

def test_start_or_end_is_wall():
    """Start or end on a wall returns None."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    # start is wall
    assert astar.find_path((0,0), (1,1)) is None
    # end is wall
    assert astar.find_path((1,1), (0,0)) is None
    # both walls
    assert astar.find_path((0,0), (0,0)) is None  # start==end but wall

def test_optimality_on_weighted_grid():
    """Verify that the path has minimum total cost."""
    grid = [
        [1, 5, 5],
        [1, 5, 5],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (0, 2)
    # Direct path: (0,0)->(0,1)->(0,2) cost = 5+5 = 10
    # Alternative: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2) cost = 1+1+1+1+5+5 = 14
    # So direct is optimal. But we want a case where detour is cheaper.
    # Let's use a grid where direct path has high cost and detour has low cost.
    grid2 = [
        [1, 10, 10],
        [1, 10, 10],
        [1,  1,  1]
    ]
    astar2 = AStarGrid(grid2)
    start2 = (0,0)
    end2 = (0,2)
    # Direct: 10+10=20
    # Detour via bottom: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2) cost = 1+1+1+1+10+10 = 24? Actually (1,2) and (0,2) are 10 each, so 24. Still direct cheaper.
    # Let's make the direct path even more expensive: use 100.
    grid3 = [
        [1, 100, 100],
        [1, 100, 100],
        [1,   1,   1]
    ]
    astar3 = AStarGrid(grid3)
    start3 = (0,0)
    end3 = (0,2)
    # Direct: 100+100=200
    # Detour: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2) cost = 1+1+1+1+100+100 = 204. Still direct cheaper.
    # Actually we need a detour that avoids the high cost cells entirely. For start (0,0) and end (0,2), the only way to avoid the high cost cells on the top row is to go around via bottom, but the bottom row cells (1,2) and (0,2) are still high cost. So we need to design a grid where the high cost cells are only on the direct path and the detour uses low cost cells.
    # Example: start (0,0), end (2,2). Direct path through (1,1) costs 100, but detour via bottom row costs 1 each.
    grid4 = [
        [1, 1, 1],
        [1, 100, 1],
        [1, 1, 1]
    ]
    astar4 = AStarGrid(grid4)
    start4 = (0,0)
    end4 = (2,2)
    path4 = astar4.find_path(start4, end4)
    assert path4 is not None
    # Optimal path should avoid (1,1) because it's 100. So it should go around via (0,1)->(0,2)->(1,2)->(2,2) or similar.
    # Check that path does not include (1,1)
    assert (1,1) not in path4
    total_cost = sum(grid4[r][c] for (r,c) in path4[1:])
    # Expected optimal cost: 4 (since 4 steps of cost 1)
    assert total_cost == 4

# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__])
```