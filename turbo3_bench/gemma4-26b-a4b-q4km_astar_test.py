import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A class representing a 2D grid for A* pathfinding.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the AStarGrid with a movement cost grid.

        :param grid: A 2D list where 0 is an impassable wall and positive ints are movement costs.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculates the Manhattan distance between two points.

        :param a: Starting coordinate (row, col).
        :param b: Target coordinate (row, and col).
        :return: Manhattan distance.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from start to end using the A* algorithm.

        :param start: Starting (row, col) coordinate.
        :param end: Target (row, col) coordinate.
        :return: List of (row, col) coordinates representing the path, or None if no path exists.
        :raises ValueError: If start or end coordinates are out of grid bounds.
        """
        # 1. Bounds Check
        for r, c in [start, end]:
            if not (0 <= r < self.rows and 0 <= c < self.s_cols if hasattr(self, 's_cols') else 0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError("Start or end coordinates are out of bounds.")

        # 2. Wall Check
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # 3. Start equals End
        if start == end:
            return [start]

        # A* Data Structures
        # open_set stores (f_score, current_node)
        open_set = []
        heapq.heappush(open, (self._heuristic(start, end), start))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score[node] is the cost of the cheapest path from start to node currently known.
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        
        # f_score[node] = g_score[node] + heuristic(node, end)
        f_score: Dict[Tuple[int, int], float] = {start: self._heuristic(start, end)}

        # Directions: Up, Down, Left, Right
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_set:
            # Pop node with the lowest f_score
            _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)

                # Check bounds and walls
                if 0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols:
                    cost = self.grid[neighbor[0]][neighbor[1]]
                    if cost == 0:
                        continue
                    
                    # Tentative g_score is the distance from start to the neighbor through current
                    tentative_g_score = g_score[current] + cost

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        # This path to neighbor is better than any previous one. Record it!
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                        
                        # Add to open set if not already there (or if we found a better path)
                        # Using a simple approach: push to heap. The heap will handle the lowest f_score.
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

# --- Pytest Suite ---
import pytest

def get_path_cost(grid, path):
    """Helper to calculate total cost of a path (sum of costs of all cells except start)."""
    if not path: return 0
    cost = 0
    for i in range(1, len(path)):
        r, c = path[i]
        cost += grid[path[i-1][0]][path[i-1][1]] # This is wrong logic, let's use:
    # Correct logic: cost is the sum of the values of the cells entered.
    cost = 0
    for i in range(1, len(path)):
        r, c = path[i]
        cost += grid[r][c]
    return cost

def test_simple_path():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Shortest path in uniform grid is 4 steps (5 nodes total)
    assert len(path) == 5
    assert get_path_cost(grid, path) == 4

def test_path_around_obstacles():
    # 0 is wall. Path must go around the wall at (1,1)
    grid = [[1, 1, 1], 
            [1, 0, 1], 
            [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Path must be (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) or similar
    assert (1, 1) not in path
    assert get_path_cost(grid, path) == 4

def test_weighted_grid():
    # Path should prefer the longer route with cost 1 over the short route with cost 10
    grid = [[1, 10, 1],
            [1, 10, 1],
            [1, 1,  1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    # Path should go (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2) -> (1,2) -> (0,2)
    # Total cost: 1+1+1+1+1+1 = 6. Direct path cost would be 10+1 = 11.
    assert get_path_cost(grid, path) == 6
    assert (0, 1) not in path

def test_no_path_exists():
    grid = [[1, 0, 1],
            [1, 0, 1],
            [1, 0, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_invalid_coordinates():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_lamp_path = astar.find_path((-1, 0), (1, 1)) # Out of bounds
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5)) # Out of bounds

def test_start_is_wall():
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (1, 1)) is None

if __name__ == "__main__":
    # This allows running the file directly to execute tests
    pytest.main([__file__])