import heapq
from typing import List, Tuple, Optional, Dict, Set

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initializes the grid. 
        grid[r][c] > 0 is the cost to enter the cell. 
        grid[r][c] == 0 is an impassable wall.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path using A* algorithm.
        Returns a list of (r, c) coordinates or None if no path exists.
        """
        # Bounds and Wall checks
        for r, c in [start, end]:
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError("Coordinates out of bounds")
            if self.grid[r][c] == 0:
                return None

        if start == end:
            return [start]

        # Priority Queue: (priority, current_cost, current_node)
        # priority = g_score + heuristic
        open_set = [(self._heuristic(start, end), 0, start)]
        
        # Tracking costs and parents
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        
        # To avoid re-processing nodes with higher costs
        visited_costs: Dict[Tuple[int, int], int] = {start: 0}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            # Optimization: if we found a better way to this node already, skip
            if current_g > g_score.get(current, float('inf')):
                continue

            r, c = current
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-directional
                neighbor = (r + dr, c + dc)
                nr, nc = neighbor

                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    weight = self.grid[nr][nc]
                    if weight == 0:  # Wall
                        continue
                    
                    tentative_g = current_g + weight
                    
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        g_score[neighbor] = tentative_g
                        priority = tentative_g + self._heuristic(neighbor, end)
                        heapq.heappush(open_set, (priority, tentative_g, neighbor))
                        came_from[neighbor] = current
                        
        return None

# --- Pytest Tests ---
import pytest

def test_simple_path():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path == [(0, 0), (0, 1), (0, 2)]
    # Cost: 0 (start) + 1 (enter 0,1) + 1 (enter 0,2) = 2
    # Note: The problem defines cost to *enter* a cell. 
    # Total cost = sum of grid[r][c] for all cells in path except start.
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 2

def test_path_around_obstacles():
    # 0 is wall
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 1), (2, 1))
    # Must go around the wall at (1, 1)
    assert (1, 1) not in path
    assert len(path) == 5 # (0,1)->(0,0)->(1,0)->(2,0)->(2,1) or similar
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4

def test_weighted_grid_optimality():
    # Path A: (0,0)->(0,1)->(0,2) cost: 1+1 = 2
    # Path B: (0,0)->(1,0)->(1,1)->(1,2)->(0,2) cost: 1+1+1+1 = 4
    # Path C: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2) cost: 1+1+1+1+1+1 = 6
    # But let's make a high cost path:
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1,  1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    # Should go down and around the 10s: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2)
    # Cost: 1+1+1+1+1+1 = 6. The direct path cost is 10+1=11.
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 6

def test_no_path_exists():
    grid = [
        [1, 0, 1],
        [0, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_invalid_coordinates():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))
    # Start/End is wall
    grid_wall = [[1, 0], [1, 1]]
    astar_wall = AStarGrid(grid_wall)
    assert astar_wall.find_path((0, 1), (1, 1)) is None