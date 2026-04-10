import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A class to perform A* pathfinding on a 2D grid with weighted movement costs.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the grid.
        :param grid: 2D list where 0 is a wall and >0 is the cost to enter the cell.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, r: int, c: int) -> bool:
        """Checks if a coordinate is within grid bounds and is not a wall."""
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] > 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculates Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using the A* algorithm.
        :return: List of (row, col) tuples or None if no path exists.
        :raises ValueError: If start or end coordinates are out of bounds.
        """
        # Check bounds
        for r, c in [start, end]:
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(f"Coordinate ({r}, {c}) is out of bounds.")

        # Handle start/end being walls
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Handle start == end
        if start == end:
            return [start]

        # Priority Queue: (f_score, current_node)
        open_set: List[Tuple[int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, start))

        # Tracking dictionaries
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        
        # Set for O(1) lookup of nodes in open_set
        open_set_hash = {start}

        while open_set:
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            r, c = current
            # 4-directional movement
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (r + dr, c + dc)

                if self._is_valid(neighbor[0], neighbor[1]):
                    # Cost to enter neighbor = current g_score + neighbor's weight
                    tentative_g_score = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._heuristic(neighbor, end)
                        
                        if neighbor not in open_set_hash:
                            heapq.heappush(open_set, (f_score, neighbor))
                            open_set_hash.add(neighbor)

        return None

# --- Tests ---

import pytest

def test_simple_path():
    """Test a simple path on a uniform grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Total cost: 1 (start) + 1 + 1 + 1 + 1 = 5 (Wait, cost is to *enter* cell)
    # Path: (0,0)->(0,1)->(0,2)->(1,2)->(2,2). Costs: enter (0,1)=1, (0,2)=1, (1,2)=1, (2,2)=1.
    # Total cost calculation: sum(grid[r][c] for r,c in path[1:])
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4

def test_path_around_obstacles():
    """Test pathfinding around a wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 0))
    # Must go around the 0 at (1, 1)
    assert (1, 0) in path or (0, 1) in path # Path must exist
    assert path is not None
    # The path (0,0)->(1,0)->(2,0) is blocked by (1,1) being a wall? 
    # No, (1,0) is not a wall. (1,1) is.
    # Path (0,0)->(1,0)->(2,0) is valid and cost is 1+1=2.
    assert sum(grid[r][c] for r, c in path[1:]) == 2

def test_weighted_grid():
    """Test that path prefers lower-cost cells."""
    grid = [
        [1, 1, 1],
        [1, 9, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Direct path (0,0)->(1,0)->(2,0) cost 2.
    # Path (0,0)->(0,1)->(1,1)->(2,1)->(2,0) cost 1+9+1+1 = 12.
    # Path (0,0)->(0,1)->(0,2)->(1,2)->(2,2)->(2,1)->(2,0) cost 1+1+1+1+1+1 = 6.
    # The shortest path should be (0,0)->(1,0)->(2,0) cost 2.
    # Let's make the direct path expensive.
    grid_v2 = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar_v2 = AStarGrid(grid_v2)
    path = astar_v2.find_path((0, 0), (0, 2))
    # Should go (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2)
    # Cost: 1+1+1+1+1+1 = 6.
    # Direct path (0,0)->(0,1)->(0,2) cost: 10+1 = 11.
    total_cost = sum(grid_v2[r][c] for r, c in path[1:])
    assert total_cost == 6

def test_no_path_exists():
    """Test when the destination is fully blocked."""
    grid = [
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_start_equals_end():
    """Test when start and end are the same."""
    grid = [[5]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_invalid_coordinates():
    """Test that out of bounds raises ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))