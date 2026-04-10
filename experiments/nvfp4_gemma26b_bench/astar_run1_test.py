import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A class representing a 2D grid for A* pathfinding.
    The grid contains weights where 0 represents a wall and 1 or more represents traversable cost.
    """

    def __init__(self, grid: List[List[float]]):
        """
        Initialize the grid.
        :param grid: A 2D list of numbers. 0 is a wall, >0 is the movement cost.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Returns valid 4-directional neighbors (up, down, left, right)."""
        r, c = pos
        neighbors = []
        # Directions: Up, Down, Left, Right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] > 0:  # 0 is a wall
                    neighbors.append((nr, nc))
        return neighbors

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using A* algorithm.
        
        :param start: (row, col) tuple
        :param end: (row, col) tuple
        :return: List of (row, col) tuples representing the path, or None if no path exists.
        :raises ValueError: If start or end are out of bounds or on a wall.
        """
        # Bounds and Wall Validation
        for point in [start, end]:
            if not (0 <= point[0] < self.rows and 0 <= point[1] < self.cols):
                raise ValueError(f"Point {point} is out of bounds.")
            if self.grid[point[0]][point[1]] == 0:
                raise ValueError(f"Point {point} is a wall.")

        if start == end:
            return [start]

        # Priority Queue: (priority, current_cost, current_pos)
        # priority = current_cost + heuristic
        pq = [(0.0, 0.0, start)]
        
        # tracking: cost_so_far[pos] = total_cost_to_reach_pos
        cost_so_far: Dict[Tuple[int, int], float] = {start: 0.0}
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        while pq:
            _, current_cost, current = heapq.heappop(pq)

            if current == end:
                break

            # If we found a better path to this node already, skip
            if current_cost > cost_so_far.get(current, float('inf')):
                continue

            for neighbor in self._get_neighbors(current):
                # Weight of the neighbor cell
                weight = self.grid[neighbor[0]][neighbor[1]]
                new_cost = current_cost + weight

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + self._heuristic(neighbor, end)
                    heapq.heappush(pq, (priority, new_cost, neighbor))
                    came_from[neighbor] = current

        # Reconstruct path
        if end not in came_from:
            return None

        path = []
        curr = end
        while curr is not None:
            path.append(curr)
            curr = came_from[curr]
        return path[::-1]

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
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_no_path_due_to_walls():
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # (0,0) cannot reach (0,2) because of the wall at (0,1) and (1,1)
    assert astar.find_path((0, 0), (0, 2)) is None

def test_weighted_path_preference():
    # Path through (0,1) is cost 10, path through (1,0)->(2,0) is cost 2
    grid = [
        [1, 10, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    # Should avoid the 10 weight cell
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
    # Manual test run if not using pytest command
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    solver = AStarGrid(grid)
    print(f"Path: {solver.find_path((0,0), (2,2))}")