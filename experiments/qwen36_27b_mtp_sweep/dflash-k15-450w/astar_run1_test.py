import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid representation:
    - 0: Wall (impassable)
    - >0: Traversal cost to enter the cell
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the pathfinding grid.
        
        :param grid: 2D list of integers representing cell costs. 
                     Must be rectangular. 0 denotes walls.
        :raises ValueError: If grid is empty or non-rectangular.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        if any(len(row) != len(grid[0]) for row in grid):
            raise ValueError("Grid must be rectangular.")
            
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* search.
        
        :param start: (row, col) tuple for the starting position.
        :param end: (row, col) tuple for the target position.
        :return: List of (row, col) tuples representing the optimal path, 
                 or None if no valid path exists.
        :raises ValueError: If start or end coordinates are out of bounds.
        """
        # Boundary validation
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds.")

        # Immediate success case
        if start == end:
            return [start]

        # Priority queue: (f_score, tie_breaker, position)
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))
        
        g_score: dict = {start: 0}
        came_from: dict = {}
        closed_set: set = set()
        tie_breaker = 1

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # 4-directional neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                nr, nc = neighbor

                # Bounds & wall check
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr][nc] == 0:
                    continue
                if neighbor in closed_set:
                    continue

                move_cost = self.grid[nr][nc]
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    # Manhattan heuristic
                    h = abs(nr - end[0]) + abs(nc - end[1])
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, tie_breaker, neighbor))
                    tie_breaker += 1

        return None

import pytest

def test_basic_pathfinding():
    """Test standard path on an unweighted grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # 4 steps + start node

def test_start_equals_end():
    """Test immediate return when start and end are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_blocked_by_walls():
    """Test that None is returned when walls completely block the path."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_weighted_optimality():
    """Test that A* chooses the cheaper path over the shorter path."""
    grid = [
        [1, 100, 1],
        [1, 100, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Should avoid the expensive center column
    assert (0, 1) not in path and (1, 1) not in path
    # Verify total traversal cost matches optimal route (cost = 4)
    total_cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert total_cost == 4

def test_complex_weighted_routing():
    """Test routing around a high-cost cell in a mixed grid."""
    grid = [
        [1, 2, 1],
        [1, 1, 1],
        [1, 10, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Optimal path should bypass (2,1) which has cost 10
    assert (2, 1) not in path
    # Expected optimal cost: 1+1+1+1 = 4
    total_cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert total_cost == 4