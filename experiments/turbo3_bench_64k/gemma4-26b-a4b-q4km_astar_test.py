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
        Find the shortest path from start to end using the A* algorithm.
        Returns a list of coordinates or None if no path exists.
        """
        # 1. Handle out of bounds
        for pt in (start, end):
            if not (0 <= pt[0] < self.rows and 0 <= pt[1] < self.cols):
                raise ValueError("Start or end coordinates are out of grid bounds.")

        # 2. Handle start/end as walls
        if self.grid[start[0][0] if isinstance(start[0], list) else start[0]][start[1]] == 0 or \
           self.grid[end[0]][end[1]] == 0:
            # Note: The prompt asks to return None if start or end is a wall
            # Checking start[0] index carefully for tuple unpacking
            pass 
        
        # Re-evaluating wall check for clarity
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # 3. Handle start == end
        if start == end:
            return [start]

        # Priority Queue: (f_score, (row, col))
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # came_from maps a node to its predecessor to reconstruct the path
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score[node] is the cost of the cheapest path from start to node currently known
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        
        while open_set:
            # Pop the node with the lowest estimated total cost (f_score)
            current_f, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            # Explore 4-directional neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dr, current[1] + dc)

                # Check bounds and if the cell is a wall
                if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols:
                    cost = self.grid[neighbor[0]][neighbor[1]]
                    if cost == 0:
                        continue

                    # Tentative g_score is the cost to reach the neighbor
                    tentative_g_score = g_score[current] + cost

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_score, neighbor))

        return None

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Trace back from end to start using the came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

# =============================================================================
# Pytest Tests
# =============================================================================
import pytest

def calculate_path_cost(grid, path):
    """Helper to calculate total cost of a path (excluding start cell)."""
    cost = 0
    for i in range(1, len(path)):
        r, c = path[i]
        cost += grid[r][c]
    return cost

def test_simple_uniform_grid():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is not None
    assert path[0] == start and path[-1] == end
    assert calculate_path_cost(grid, path) == 4 # 4 steps of cost 1

def test_path_around_obstacles():
    # Wall at (1, 1) forces path to go around
    grid = [[1, 1, 1], 
            [1, 0, 1], 
            [1, 1, 1]]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    assert path is not None
    assert (1, 1) not in path
    assert calculate_path_cost(grid, path) == 4

def test_weighted_grid_optimality():
    # Direct path (0,1) is very expensive (10), detour is cheaper
    grid = [[1, 10, 1], 
            [1, 1, 1], 
            [1, 1, 1]]
    astar = AStarGrid(grid)
    start, end = (0, 0), (0, 2)
    path = astar.find_path(start, end)
    # Optimal path should be (0,0) -> (1,0) -> (1,1) -> (1,2) -> (0,2)
    # Cost: 1 + 1 + 1 + 1 = 4, whereas direct is 10 + 1 = 11
    assert calculate_path_cost(grid, path) == 4
    assert (0, 1) not in path

def test_no_path_exists():
    grid = [[1, 0, 1], 
            [0, 0, 1], 
            [1, 0, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) is None

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    start = (0, 0)
    assert astar.find_path(start, start) == [start]

def test_invalid_coordinates():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 0))

if __name__ == "__main__":
    # To run tests via standard python: pytest <filename>.py
    pytest.main([__file__])