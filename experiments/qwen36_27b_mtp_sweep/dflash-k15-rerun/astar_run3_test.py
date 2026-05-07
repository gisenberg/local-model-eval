import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid representation:
    - 0 represents a wall (impassable)
    - Positive numbers represent the traversal cost to enter that cell
    """

    def __init__(self, grid: List[List[float]]) -> None:
        """
        Initialize the AStarGrid with a 2D grid.
        
        Args:
            grid: 2D list where 0 is a wall and positive values are traversal costs.
            
        Raises:
            ValueError: If grid is empty or rows have unequal lengths.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        if any(len(row) != self.cols for row in grid):
            raise ValueError("Grid rows must have equal length")

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if coordinates are within grid bounds."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if coordinates represent a wall."""
        return self.grid[pos[0]][pos[1]] == 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic for 4-directional movement."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: Starting coordinates (row, col)
            end: Target coordinates (row, col)
            
        Returns:
            List of coordinates representing the optimal path, or None if unreachable.
            
        Raises:
            ValueError: If start/end are out of bounds or positioned on a wall.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds")
        if self._is_wall(start) or self._is_wall(end):
            raise ValueError("Start or end position is a wall")

        if start == end:
            return [start]

        # Priority queue: (f_score, tie_breaker_counter, position)
        counter = 0
        open_set = [(self._heuristic(start, end), counter, start)]
        
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            # 4-directional neighbors
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)

                if not self._is_valid(neighbor) or self._is_wall(neighbor):
                    continue

                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return None

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Backtrack from end to start using came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


# ========================
# PYTEST TESTS
# ========================
import pytest

def test_basic_pathfinding():
    """Test standard pathfinding on a simple grid with forced route."""
    grid = [
        [1, 1, 0],
        [0, 1, 0],
        [0, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Optimal length

def test_start_equals_end():
    """Test when start and end coordinates are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_exists():
    """Test when end is completely surrounded by walls."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_at_start_end_raises_value_error():
    """Test that starting or ending on a wall raises ValueError."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((1, 1), (0, 0))

def test_weighted_optimal_path():
    """Test that A* chooses lower cost path over shorter geometric path."""
    # Direct path cost: 10 + 1 = 11
    # Longer path cost: 1 + 1 + 1 + 1 = 4
    grid = [
        [1, 10, 1],
        [1,  1, 1],
        [1,  1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (0, 2)
    
    # Verify it took the cheaper route through row 1
    assert (1, 0) in path and (1, 1) in path and (1, 2) in path
    
    # Calculate total traversal cost (excluding start cell)
    cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert cost == 4