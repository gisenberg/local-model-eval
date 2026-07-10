# astar_grid.py
from typing import List, Tuple, Optional
import heapq
import math

Coordinate = Tuple[int, int]


class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.

    Grid representation:
    - 0 represents a wall (impassable)
    - Positive integers represent the cost to enter that cell
    - Movement is 4-directional (up, down, left, right)

    Attributes:
        grid: 2D list of non-negative integers. 0 = wall, >0 = movement cost.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid with the given 2D grid.

        Args:
            grid: 2D list of non-negative integers. 0 = wall, >0 = movement cost.

        Raises:
            ValueError: If grid is empty or rectangularity is violated.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

        # Ensure grid is rectangular
        for row in grid:
            if len(row) != self.cols:
                raise ValueError("Grid must be rectangular")

    def _is_valid(self, x: int, y: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= x < self.rows and 0 <= y < self.cols

    def _is_wall(self, x: int, y: int) -> bool:
        """Check if a cell is a wall."""
        return self.grid[x][y] == 0

    def _get_neighbors(self, x: int, y: int) -> List[Coordinate]:
        """Get valid (non-wall) neighbors in 4 directions."""
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self._is_valid(nx, ny) and not self._is_wall(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def _heuristic(self, a: Coordinate, b: Coordinate) -> float:
        """
        Manhattan distance heuristic.

        Args:
            a: First coordinate (x1, y1)
            b: Second coordinate (x2, y2)

        Returns:
            Manhattan distance between a and b
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Coordinate, end: Coordinate) -> Optional[List[Coordinate]]:
        """
        Find optimal path from start to end using A* algorithm.

        Args:
            start: Starting coordinate (x1, y1)
            end: Target coordinate (x2, y2)

        Returns:
            List of coordinates representing the optimal path (including start and end),
            or None if no path exists.

        Raises:
            ValueError: If start or end coordinates are out of bounds or are walls.
        """
        # Validate start and end
        for coord, name in [(start, "start"), (end, "end")]:
            x, y = coord
            if not self._is_valid(x, y):
                raise ValueError(f"{name} coordinate ({x}, {y}) is out of bounds")
            if self._is_wall(x, y):
                raise ValueError(f"{name} coordinate ({x}, {y}) is a wall")

        # Handle same start and end
        if start == end:
            return [start]

        # A* algorithm data structures
        # f_score = g_score + heuristic
        open_set = []  # heap of (f_score, g_score, (x, y))
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))

        came_from = {}  # coordinate -> previous coordinate
        g_score = {start: 0}

        while open_set:
            _, current_g, current = heapq.heappop(open_set)

            # Skip if we've found a better path to this node already
            if current_g > g_score.get(current, float('inf')):
                continue

            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]  # Reverse to get start -> end

            # Process neighbors
            for neighbor in self._get_neighbors(*current):
                # Cost to move to neighbor is the neighbor's grid value
                # (cost to enter that cell)
                tentative_g = current_g + self.grid[neighbor[0]][neighbor[1]]

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))

        # No path found
        return None

# test_astar_grid.py
import pytest


class TestAStarGrid:
    """Tests for AStarGrid pathfinding functionality."""

    def test_simple_path(self):
        """Test basic pathfinding with no obstacles."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        # Manhattan path length = 4 steps
        assert path is not None
        assert len(path) == 5  # start + 4 moves
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)

    def test_path_with_walls(self):
        """Test pathfinding around obstacles."""
        grid = [
            [1, 1, 1],
            [1, 0, 1],  # wall in center
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        assert path is not None
        # Path must go around the wall (e.g., right-down-right-up-left-down-right)
        assert (1, 1) not in path  # wall cell is not in path

    def test_weighted_path_optimization(self):
        """Test that A* chooses lower-cost path."""
        grid = [
            [1, 100, 1],  # high cost in middle
            [1, 100, 1],
            [1, 1,   1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        assert path is not None
        # Path should avoid high-cost cells if possible
        # The optimal path avoids the middle column: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)
        # Cost = 1+1+1+1+1 = 5
        # Alternative via center would cost 1+100+1+100+1 = 203
        assert (1, 1) not in path  # verify not taking high-cost route

    def test_no_path_possible(self):
        """Test when no path exists due to walls."""
        grid = [
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (0, 2))
        assert path is None

    def test_start_equals_end(self):
        """Test when start and end are the same cell."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((1, 1), (1, 1))
        assert path == [(1, 1)]

    def test_invalid_coordinates(self):
        """Test error handling for out-of-bounds and wall coordinates."""
        grid = [
            [1, 1, 1],
            [1, 0, 1],  # wall at (1,1)
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Test out-of-bounds start
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (1, 1))
        
        # Test out-of-bounds end
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (3, 3))
        
        # Test start on wall
        with pytest.raises(ValueError):
            astar.find_path((1, 1), (0, 0))
        
        # Test end on wall
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (1, 1))

    def test_rectangular_grid(self):
        """Test with non-square grid."""
        grid = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 4))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])