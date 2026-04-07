import heapq
from typing import List, Tuple, Optional, Dict, Set

class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the grid.
        
        :param grid: A 2D list where 0 is an impassable wall and positive ints are movement costs.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculates the Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using the A* algorithm.

        :param start: (row, col) starting coordinates.
        :param end: (row, col) target coordinates.
        :return: List of (row, col) tuples representing the path, or None if no path exists.
        :raises ValueError: If start or end coordinates are out of bounds.
        """
        # 1. Bounds Check
        for r, c in [start, end]:
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError(f"Coordinates ({r}, {c}) are out of bounds.")

        # 2. Wall Check
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # 3. Start is End
        if start == end:
            return [start]

        # A* Data Structures
        # open_set stores (f_score, (row, col))
        open_set: List[Tuple[int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, start))

        # came_from maps a node to its predecessor for path reconstruction
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # g_score[node] is the cost of the cheapest path from start to node currently known
        g_score: Dict[Tuple[int, int], int] = {start: 0}

        # Set for faster lookup of nodes in the open set
        open_set_hash: Set[Tuple[int, int]] = {start}

        while open_set:
            # Pop the node with the lowest f_score
            _, current = heapq.heappop(open_set)
            open_set_hash.remove(current)

            if current == end:
                return self._reconstruct_path(came_from, current)

            curr_r, curr_c = current

            # 4-directional movement (Up, Down, Left, Right)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (curr_r + dr, curr_c + dc)
                nr, nc = neighbor

                # Check bounds and if neighbor is a wall
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    weight = self.grid[nr][nc]
                    if weight == 0:
                        continue

                    # Calculate tentative g_score
                    tentative_g_score = g_score[current] + weight

                    if tentative_g_score < g_score.get(neighbor, float('inf')):
                        # This path to neighbor is better than any previous one
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score = tentative_g_score + self._manhattan_distance(neighbor, end)
                        
                        if neighbor not in open_set_hash:
                            heapq.heappush(open_set, (f_score, neighbor))
                            open_set_hash.add(neighbor)

        return None

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstructs the path from the came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

# --- Pytest Tests ---

import pytest

def test_simple_path():
    """Test a simple path on a uniform grid."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path == [(0, 0), (0, 1), (0, 2)]
    # Cost: grid[0][1] + grid[0][2] = 1 + 1 = 2
    # We verify optimality by checking path length/logic

def test_path_around_obstacles():
    """Test finding a path around walls (0s)."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Path must go around the center wall
    path = astar.find_path((0, 1), (2, 1))
    assert path is not None
    assert (1, 1) not in path
    # Check path is valid (connected)
    for i in range(len(path) - 1):
        dist = abs(path[i][0] - path[i+1][0]) + abs(path[i][1] - path[i+1][1])
        assert dist == 1

def test_weighted_grid():
    """Test that the path prefers lower-cost cells."""
    grid = [
        [1, 1, 1],
        [1, 9, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Direct path (0,0) -> (1,1) -> (2,2) would cost 9 + 1 = 10
    # Path around (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) costs 1+1+1+1 = 4
    path = astar.find_path((0, 0), (2, 2))
    assert (1, 1) not in path
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

def test_no_path_exists():
    """Test when the target is completely blocked."""
    grid = [
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_start_equals_end():
    """Test when start and end are the same."""
    grid = [[1, 5], [5, 5]]
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

if __name__ == "__main__":
    # This allows running the file directly with pytest
    pytest.main([__file__])