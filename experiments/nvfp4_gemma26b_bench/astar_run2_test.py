import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A class representing a 2D grid for A* pathfinding.
    The grid contains weights where a higher weight means a more 'expensive' cell to traverse.
    0 represents a wall (impassable).
    """

    def __init__(self, grid: List[List[float]]):
        """
        Initialize the grid.
        :param grid: A 2D list of floats. 0.0 is a wall, > 0.0 is a traversable weight.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Returns valid 4-directional neighbors (Up, Down, Left, Right)."""
        r, c = pos
        neighbors = []
        # Directions: Up, Down, Left, Right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                # Only return if not a wall (0)
                if self.grid[nr][nc] > 0:
                    neighbors.append((nr, nc))
        return neighbors

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using A* algorithm.
        
        :param start: (row, col) starting position.
        :param end: (row, col) target position.
        :return: List of (row, col) tuples representing the path, or None if no path exists.
        :raises ValueError: If start or end are out of bounds or on a wall.
        """
        # Bounds and Wall Validation
        for pos in [start, end]:
            if not (0 <= pos[0] < self.rows and 0 <= pos[1] < self.cols):
                raise ValueError(f"Position {pos} is out of bounds.")
            if self.grid[pos[0]][pos[1]] <= 0:
                raise ValueError(f"Position {pos} is a wall or invalid.")

        if start == end:
            return [start]

        # Priority Queue stores: (priority, current_cost, current_pos)
        # priority = current_cost + heuristic
        pq = [(0.0 + self._heuristic(start, end), 0.0, start)]
        
        # track the best cost to reach a node
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        # track the path
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        while pq:
            _, current_cost, current = heapq.heappop(pq)

            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            # If we found a better way to this node already, skip
            if current_cost > g_score.get(current, float('inf')):
                continue

            for neighbor in self._get_neighbors(current):
                # Weight of the cell we are moving INTO
                weight = self.grid[neighbor[0]][neighbor[1]]
                tentative_g_score = current_cost + weight

                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self._heuristic(neighbor, end)
                    heapq.heappush(pq, (f_score, tentative_g_score, neighbor))

        return None

# --- Pytest Tests ---
import pytest

def test_path_found():
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert (0, 0) in path
    assert (2, 2) in path

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_no_path_due_to_walls():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None

def test_weighted_path_preference():
    # Path through (0,1) is weight 10, path through (1,0) and (2,0) is weight 1
    grid = [
        [1, 10, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Should avoid the 10 weight cell if a cheaper path exists
    path = astar.find_path((0, 0), (0, 2))
    assert (0, 1) not in path

def test_out_of_bounds_raises_error():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_wall_start_raises_error():
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))

if __name__ == "__main__":
    # Manual execution if not using pytest command
    grid_example = [
        [1, 1, 1, 1],
        [1, 5, 5, 1],
        [1, 1, 1, 1]
    ]
    solver = AStarGrid(grid_example)
    print(f"Path: {solver.find_path((0,0), (2,3))}")