import heapq
from typing import List, Tuple, Optional, Dict, Set
from collections import deque

class AStarGrid:
    """A class to perform A* pathfinding on a weighted 2D grid."""

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid with movement costs.

        Args:
            grid: 2D list where each value represents movement cost (0 = impassable wall)
        """
        self.grid = grid
        self.rows = len(grid)
        if self.rows == 0:
            raise ValueError("Grid cannot be empty")
        self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.

        Args:
            start: (row, col) coordinates of the starting position
            end: (row, col) coordinates of the target position

        Returns:
            List of (row, col) coordinates representing the path, or None if no path exists
        """
        # Validate coordinates
        if not self._is_valid_coordinate(start) or not self._is_valid_coordinate(end):
            raise ValueError("Start or end coordinates are out of bounds")
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        if start == end:
            return [start]

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Priority queue: (f_score, g_score, current_node)
        open_set = []
        heapq.heappush(open_set, (0, 0, start))

        # For node n, came_from[n] is the node immediately preceding it on the cheapest path
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # For node n, g_score[n] is the cost of the cheapest path from start to n
        g_score: Dict[Tuple[int, int], int] = {start: 0}

        # For node n, f_score[n] = g_score[n] + h(n)
        f_score: Dict[Tuple[int, int], int] = {start: self._manhattan_distance(start, end)}

        open_set_hash: Set[Tuple[int, int]] = {start}

        while open_set:
            current = heapq.heappop(open_set)[2]
            open_set_hash.remove(current)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._is_valid_coordinate(neighbor):
                    continue
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue

                # tentative_g_score is the distance from start to the neighbor through current
                tentative_g_score = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._manhattan_distance(neighbor, end)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        return None  # No path found

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to current node using came_from dictionary."""
        path = []
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append(current)
        return path[::-1]  # Reverse to get start to end

    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_valid_coordinate(self, coord: Tuple[int, int]) -> bool:
        """Check if coordinates are within grid bounds."""
        row, col = coord
        return 0 <= row < self.rows and 0 <= col < self.cols

    def get_path_cost(self, path: List[Tuple[int, int]]) -> int:
        """Calculate the total cost of a path."""
        if not path:
            return 0
        cost = self.grid[path[0][0]][path[0][1]]
        for i in range(1, len(path)):
            cost += self.grid[path[i][0]][path[i][1]]
        return cost

import pytest
from typing import List, Tuple, Optional

def test_simple_path_uniform_grid():
    """Test simple path on uniform grid (all cells have same cost)."""
    grid = [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert astar.get_path_cost(path) == 5

def test_path_around_obstacles():
    """Test path that goes around obstacles."""
    grid = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (3, 4))
    assert path == [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)]
    assert astar.get_path_cost(path) == 8

def test_weighted_grid():
    """Test path that prefers lower-cost cells."""
    grid = [
        [1, 2, 1],
        [1, 1, 1],
        [1, 2, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    # Should go through the center (cost 1) rather than around (cost 2+1+2=5)
    assert path == [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)]
    assert astar.get_path_cost(path) == 5  # 1+1+1+1+1 (but actually 1+1+1+1=4, wait no - the path is 5 steps but cost is sum of grid values)
    # Correction: the path cost should be 1 (start) + 1 (middle) + 1 (end) = 3
    # Wait, no - the path is [(0,0), (1,0), (1,1), (1,2), (2,2)] with costs 1+1+1+1+1=5
    # But the optimal path is actually [(0,0), (0,1), (1,1), (2,1), (2,2)] with cost 1+2+1+2+1=7
    # Hmm, seems my test case is wrong. Let me fix it.
    # Actually, the correct optimal path is [(0,0), (1,0), (1,1), (1,2), (2,2)] with cost 1+1+1+1+1=5
    # So the test is correct as is.

def test_no_path_exists():
    """Test when no path exists (fully blocked)."""
    grid = [
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) is None

def test_start_equals_end():
    """Test when start and end are the same."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((1, 1), (1, 1)) == [(1, 1)]

def test_invalid_coordinates():
    """Test when coordinates are out of bounds."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (3, 3))
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((1, 1), (2, 2))

def test_start_or_end_is_wall():
    """Test when start or end is a wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((1, 1), (2, 2)) is None
    assert astar.find_path((0, 0), (1, 1)) is None