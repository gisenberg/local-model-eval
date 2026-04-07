import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    Implements A* pathfinding on a 2D grid where cell values represent the cost to enter that cell.
    0 represents an impassable wall.
    """

    def __init__(self, grid: List[List[int]]:
        """
        Initialize the AStarGrid with a 2D list of costs.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using the A* algorithm.
        Returns a list of coordinates or None if no path exists.
        """
        # 1. Validate bounds
        for pt in (start, end):
            if not (0 <= pt[0] < self.rows and 0 <= pt[1] < self.cols):
                raise ValueError("Start or end coordinates are out of grid bounds.")

        # 2. Handle edge cases
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        if start == end:
            return [start]

        # Priority Queue stores (f_score, (row, col))
        # f_score = g_score (actual cost) + heuristic (estimated cost to end)
        open_set = [(0, start)]
        
        # g_score[node] is the cost of the cheapest path from start to node currently known
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        
        # came_from stores the parent of each node for path reconstruction
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_set:
            # Pop the node with the lowest f_score
            current_f, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Explore 4-directional neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                # Check bounds and if the cell is a wall
                if (0 <= neighbor[0] < self.rows and 
                    0 <= neighbor[1] < self.cols and 
                    self.grid[neighbor[0]][neighbor[1]] != 0):
                    
                    # Cost to enter the neighbor cell
                    tentative_g_score = g_score[current] + self.grid[neighbor[0]][neighbor[1]]
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score, neighbor))

        return None

# --- Pytest Tests ---

import pytest

def calculate_path_cost(grid, path):
    """Helper to calculate total cost of a path (excluding start cell)."""
    return sum(grid[r][c] for r, c in path[1:])

def test_simple_uniform_grid():
    grid = [[1, 1, 1], 
            [1, 1, 1], 
            [1, 1, 1]]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is not None
    assert len(path) == 5 # 0,0 -> 0,1 -> 0,2 -> 1,2 -> 2,2 (or similar)
    assert calculate_path_cost(grid, path) == 4

def test_path_around_obstacles():
    grid = [[1, 1, 1], 
            [1, 0, 1], 
            [1, 1, 1]]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is not None
    assert (1, 1) not in path
    assert len(path) == 5

def test_weighted_grid_optimality():
    # Direct path (0,0 -> 0,1 -> 0,2) cost is 10.
    # Detour (0,0 -> 1,0 -> 1,1 -> 1,2 -> 0,2) cost is 1+1+1+1 = 4.
    grid = [[1, 10, 1], 
            [1, 1, 1]]
    astar = AStarGrid(grid)
    start, end = (0, 0), (0, 2)
    path = astar.find_path(start, end)
    assert path == [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]
    assert calculate_path_cost(grid, path) == 4

def test_no_path_exists():
    grid = [[1, 0, 1], 
            [0, 0, 1], 
            [1, 1, 1]]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is None

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    start = (0, 0)
    path = astar.find_path(start, start)
    assert path == [start]

def test_invalid_coordinates():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))

if __name__ == "__main__":
    # To run tests without pytest CLI, you can use:
    # pytest this_file.py
    import sys
    pytest.main(sys.argv)