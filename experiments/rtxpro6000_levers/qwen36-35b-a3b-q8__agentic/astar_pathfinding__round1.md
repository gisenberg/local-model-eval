```python
import heapq
from typing import List, Tuple, Optional, Dict
import pytest


class AStarGrid:
    """A* pathfinding implementation on a weighted 2D grid."""

    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the grid with movement costs.

        Args:
            grid: 2D list where 0 represents a wall and positive integers represent entry cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """Calculate Manhattan distance between two coordinates."""
        return abs(r1 - r2) + abs(c1 - c2)

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find the shortest path from start to end using A* algorithm.

        Args:
            start: Tuple (row, col) of the starting position.
            end: Tuple (row, col) of the destination position.

        Returns:
            List of (row, col) tuples representing the optimal path, or None if no path exists.

        Raises:
            ValueError: If start or end coordinates are out of grid bounds.
        """
        sr, sc = start
        er, ec = end

        # Validate bounds
        if not (0 <= sr < self.rows and 0 <= sc < self.cols):
            raise ValueError(f"Start coordinates {start} are out of bounds.")
        if not (0 <= er < self.rows and 0 <= ec < self.cols):
            raise ValueError(f"End coordinates {end} are out of bounds.")
        
        # Check for walls
        if self.grid[sr][sc] == 0 or self.grid[er][ec] == 0:
            return None
        
        # Start equals end
        if start == end:
            return [start]

        # A* initialization
        open_set: List[Tuple[int, int, int, int, int]] = []
        heapq.heappush(open_set, (self._heuristic(sr, sc, er, ec), 0, 0, sr, sc))

        g_score: Dict[Tuple[int, int], int] = {(sr, sc): 0}
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        counter = 0  # Tie-breaker for heap

        while open_set:
            f, _, g, r, c = heapq.heappop(open_set)

            # Lazy deletion: skip if we found a better path to this node already
            if g > g_score.get((r, c), float('inf')):
                continue

            if (r, c) == end:
                # Reconstruct path
                path = []
                curr: Optional[Tuple[int, int]] = end
                while curr is not None:
                    path.append(curr)
                    curr = parent.get(curr)
                return path[::-1]

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] > 0:
                    move_cost = self.grid[nr][nc]
                    tentative_g = g + move_cost

                    if tentative_g < g_score.get((nr, nc), float('inf')):
                        parent[(nr, nc)] = (r, c)
                        g_score[(nr, nc)] = tentative_g
                        f_new = tentative_g + self._heuristic(nr, nc, er, ec)
                        counter += 1
                        heapq.heappush(open_set, (f_new, counter, tentative_g, nr, nc))

        return None


def calculate_path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Calculate total movement cost for a path (excluding start cell)."""
    return sum(grid[r][c] for r, c in path[1:])


class TestAStarGrid:
    def test_simple_path_uniform_grid(self):
        """Test simple path on a uniform grid where all cells cost 1."""
        grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is not None
        assert path[0] == (0, 0) and path[-1] == (2, 2)
        
        # Validate adjacency (4-directional movement)
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            assert abs(r1 - r2) + abs(c1 - c2) == 1
            
        # Optimal cost: 4 steps of cost 1
        assert calculate_path_cost(grid, path) == 4

    def test_path_around_obstacles(self):
        """Test pathfinding around a 2x2 wall block."""
        grid = [[1, 1, 1, 1], 
                [1, 0, 0, 1], 
                [1, 1, 1, 1]]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 3))
        
        assert path is not None
        # Must navigate around the wall: 5 cells entered, all cost 1
        assert calculate_path_cost(grid, path) == 5

    def test_weighted_grid_prefers_low_cost(self):
        """Test that A* prefers lower-cost cells over shorter geometric paths."""
        grid = [[1, 5, 1], [1, 1, 1], [1, 5, 1]]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (0, 2))
        
        assert path is not None
        # Direct top path costs 5+1=6. Middle path costs 1+1+1+1=4.
        assert calculate_path_cost(grid, path) == 4

    def test_no_path_exists(self):
        """Test fully blocked grid where no path exists."""
        grid = [[1, 1, 1], [0, 0, 0], [1, 1, 1]]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is None

    def test_start_equals_end(self):
        """Test edge case where start and end coordinates are identical."""
        grid = [[1, 1], [1, 1]]
        astar = AStarGrid(grid)
        path = astar.find_path((1, 1), (1, 1))
        
        assert path == [(1, 1)]
        assert calculate_path_cost(grid, path) == 0

    def test_invalid_coordinates_and_walls(self):
        """Test out-of-bounds coordinates and start/end on walls."""
        grid = [[1, 1], [1, 1]]
        astar = AStarGrid(grid)
        
        # Out of bounds should raise ValueError
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (0, 1))
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (2, 2))
            
        # Start or end on a wall should return None
        grid_wall = [[0, 1], [1, 1]]
        astar_wall = AStarGrid(grid_wall)
        assert astar_wall.find_path((0, 0), (1, 1)) is None
        assert astar_wall.find_path((1, 0), (0, 0)) is None
```