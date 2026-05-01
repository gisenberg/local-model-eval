import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid representation:
        - 0: Wall (impassable)
        - >0: Traversal cost to enter the cell
    """
    def __init__(self, grid: List[List[float]]):
        """
        Initialize the grid for pathfinding.
        
        Args:
            grid: 2D list where 0 represents walls and positive numbers represent costs.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0])
        self.directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two (x, y) points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: (x, y) tuple representing the starting coordinates.
            end: (x, y) tuple representing the target coordinates.
            
        Returns:
            A list of (x, y) tuples representing the optimal path, or None if unreachable.
            
        Raises:
            ValueError: If start or end is out of bounds or located on a wall.
        """
        # Validate start and end positions
        for pos, name in [(start, "start"), (end, "end")]:
            x, y = pos
            if not (0 <= x < self.width and 0 <= y < self.height):
                raise ValueError(f"{name} is out of bounds")
            if self.grid[y][x] == 0:
                raise ValueError(f"{name} is on a wall")

        if start == end:
            return [start]

        counter = 0
        # Priority queue stores (f_score, counter, (x, y))
        # Counter breaks ties deterministically when f_scores are equal
        open_set = [(self._manhattan(start, end), counter, start)]
        
        g_score = {start: 0.0}      # Cost from start to current node
        came_from = {start: None}   # Parent pointers for path reconstruction
        closed_set = set()          # Nodes already evaluated

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            cx, cy = current
            for dx, dy in self.directions:
                nx, ny = cx + dx, cy + dy

                # Bounds and wall checks
                if not (0 <= nx < self.width and 0 <= ny < self.height):
                    continue
                if self.grid[ny][nx] == 0:
                    continue

                tentative_g = g_score[current] + self.grid[ny][nx]
                
                # If we found a cheaper path to neighbor, update it
                if (nx, ny) not in g_score or tentative_g < g_score[(nx, ny)]:
                    came_from[(nx, ny)] = current
                    g_score[(nx, ny)] = tentative_g
                    f_score = tentative_g + self._manhattan((nx, ny), end)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, (nx, ny)))

        return None  # No path found

import pytest

def test_basic_pathfinding():
    """Test optimal path on a uniform grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Manhattan distance + 1 node

def test_start_equals_end():
    """Test when start and end coordinates are identical."""
    grid = [[1]]
    assert AStarGrid(grid).find_path((0, 0), (0, 0)) == [(0, 0)]

def test_no_path_exists():
    """Test unreachable target surrounded by walls."""
    grid = [[1, 0, 1], [1, 0, 1], [1, 1, 1]]
    assert AStarGrid(grid).find_path((0, 0), (2, 2)) is None

def test_out_of_bounds_raises():
    """Test ValueError for coordinates outside grid dimensions."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_at_start_end_raises():
    """Test ValueError when start or end is placed on a wall."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="on a wall"):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError, match="on a wall"):
        astar.find_path((0, 1), (0, 0))

def test_weighted_optimal_path():
    """Test that A* chooses lowest cost path, not shortest step count."""
    # High cost in middle column forces path around the bottom
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert path is not None
    # Optimal cost: 1+1+1+1 = 4 (avoids the 10s)
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]