import heapq
from typing import List, Tuple, Optional, Dict, Set

class AStarGrid:
    """
    A class to perform A* pathfinding on a 2D weighted grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the AStarGrid with a movement cost grid.

        :param grid: A 2D list where 0 represents a wall and positive integers 
                     represent the cost to enter that cell.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculates the Manhattan distance between two points.

        :param a: The first coordinate (row, col).
        :param b: The second coordinate (row, col).
        :return: The Manhattan distance.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using the A* algorithm.

        :param start: The starting (row, col) coordinate.
        :param end: The destination (row, and col) coordinate.
        :return: A list of (row, col) coordinates representing the path, 
                  or None if no path exists.
        :raises ValueError: If start or end coordinates are out of grid bounds.
        """
        # 1. Bounds Check
        for point in [start, end]:
            if not (0 <= point[0] < self.rows and 0 <= point[1] < self.cols):
                raise ValueError(f"Coordinate {point} is out of bounds.")

        # 2. Wall Check
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # 3. Edge Case: Start is End
        if start == end:
            return [start]

        # 4. A* Initialization
        # open_set stores (f_score, current_node)
        open_set: List[Tuple[int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, start))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score[node] is the cost of the cheapest path from start to node currently known.
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        
        # f_score[node] = g_score[node] + h(node).
        f_score: Dict[Tuple[int, int], float] = {start: float(self._heuristic(start, end))}

        visited: Set[Tuple[int, int]] = set()

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            visited.add(current)

            # 4-directional movement: Up, Down, Left, Right
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dr, current[1] + dc)

                # Check bounds and if it's a wall
                if (0 <= neighbor[0] < self.rows and 
                    0 <= neighbor[1] < self.cols and 
                    self.grid[neighbor[0]][neighbor[1]] > 0):
                    
                    if neighbor in visited:
                        continue

                    # Cost to enter the neighbor cell
                    tentative_g_score = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        # This path to neighbor is better than any previous one. Record it!
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                        
                        # If neighbor not in open_set (or we found a better path), push to heap
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

# --- Pytest Suite ---
import pytest

def test_simple_path():
    """Test a direct path on a uniform grid."""
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path == [(0, 0), (0, 1), (0, 2)]
    # Total cost: entering (0,1) + entering (0,2) = 1 + 1 = 2

def test_path_around_obstacles():
    """Test pathfinding when a wall blocks the direct route."""
    grid = [[1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 0))
    # Must go around the wall at (1,1)
    assert (1, 1) not in path
    assert path == [(0, 0), (1, 0), (2, 0)]

def test_weighted_grid_optimality():
    """Test that the algorithm prefers a longer path with lower total cost."""
    # Path A: (0,0)->(0,1)->(0,2) cost: 10 + 10 = 20
    # Path B: (0,0)->(1,0)->(1,1)->(1,2)->(0,2) cost: 1 + 1 + 1 + 1 = 4
    grid = [[1, 10, 10],
            [1,  1,  1],
            [1,  1,  1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    # The optimal path should go down and around the high-cost cells
    expected_path = [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]
    assert path == expected_path
    # Verify cost: 1 (to 1,0) + 1 (to 1,1) + 1 (to 1,2) + 10 (to 0,2) = 13
    # Wait, let's re-calculate: 
    # (0,0) -> (1,0) [cost 1]
    # (1,0) -> (1,1) [cost 1]
    # (1,1) -> (1,2) [cost 1]
    # (1,2) -> (0,2) [cost 10]
    # Total = 13. Direct path (0,0)->(0,1)->(0,2) is 10+10=20. 13 < 20.

def test_no_path_exists():
    """Test that None is returned when the destination is unreachable."""
    grid = [[1, 1, 1],
            [1, 0, 0],
            [1, 0, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_start_equals_end():
    """Test that path is just the start node if start == end."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_invalid_coordinates():
    """Test that ValueError is raised for out of bounds coordinates."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_start_or_end_is_wall():
    """Test that None is returned if start or end is a wall."""
    grid = [[1, 0, 1],
            [1, 1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 1), (1, 1)) is None
    assert astar.find_path((0, 0), (0, 1)) is None