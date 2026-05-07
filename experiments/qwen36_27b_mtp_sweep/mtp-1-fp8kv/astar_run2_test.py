import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid representation:
    - 0: Wall (impassable)
    - >0: Passable cell with movement cost equal to its value.
    - Coordinates are (x, y) where x is column, y is row.
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid for pathfinding.
        
        Args:
            grid: 2D list of integers representing cell costs. 0 denotes walls.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within bounds and not a wall."""
        x, y = pos
        return 0 <= x < self.cols and 0 <= y < self.rows and self.grid[y][x] != 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic for 4-directional movement."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.
        
        Args:
            start: (x, y) starting position.
            end: (x, y) target position.
            
        Returns:
            List of (x, y) tuples representing the optimal path, or None if unreachable.
            
        Raises:
            ValueError: If start/end are out of bounds or placed on a wall.
        """
        # Validation
        if not (0 <= start[0] < self.cols and 0 <= start[1] < self.rows):
            raise ValueError("Start position is out of bounds")
        if not (0 <= end[0] < self.cols and 0 <= end[1] < self.rows):
            raise ValueError("End position is out of bounds")
        if self.grid[start[1]][start[0]] == 0:
            raise ValueError("Start position is a wall")
        if self.grid[end[1]][end[0]] == 0:
            raise ValueError("End position is a wall")

        if start == end:
            return [start]

        # Priority queue stores: (f_score, tie_breaker, g_score, position)
        open_set: List[Tuple[float, int, float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, 0, 0.0, start))
        
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        counter = 1

        while open_set:
            current_f, _, current_g, current = heapq.heappop(open_set)

            # Goal reached
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Lazy deletion: skip if we've already found a better path to this node
            if current_g > g_score[current]:
                continue

            # Explore 4-directional neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if not self._is_valid(neighbor):
                    continue

                # Cost to move is the weight of the destination cell
                move_cost = self.grid[neighbor[1]][neighbor[0]]
                tentative_g = current_g + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, counter, tentative_g, neighbor))
                    counter += 1

        return None

import pytest

def test_basic_pathfinding():
    """Test simple path on an open grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Manhattan distance is 4, so optimal path has 5 nodes
    assert len(path) == 5

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_unreachable_path():
    """Test path blocked by walls returns None."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 0))
    assert path is None

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((2, 0), (0, 0))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (-1, 0))

def test_wall_at_start_or_end_raises_value_error():
    """Test that placing start/end on a wall raises ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((1, 1), (0, 0))

def test_weighted_optimal_path():
    """Test that A* chooses lower-cost path over shorter distance."""
    # Direct path cost: 10 + 1 = 11
    # Detour path cost: 1 + 1 + 1 + 1 = 4
    grid = [
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 0))
    assert path is not None
    # Should take the bottom detour
    assert path == [(0, 0), (0, 1), (1, 1), (2, 1), (2, 0)]