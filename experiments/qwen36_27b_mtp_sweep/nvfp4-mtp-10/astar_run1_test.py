import heapq
from typing import List, Tuple, Optional
import pytest


class AStarGrid:
    """A* pathfinding algorithm on a weighted 2D grid."""

    def __init__(self, grid: List[List[float]]):
        """
        Initialize the grid.
        
        Args:
            grid: 2D list where 0 represents a wall and >0 represents traversal cost.
                  Weights must be positive for optimality guarantees.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* search.
        
        Args:
            start: (row, col) tuple for start position.
            end: (row, col) tuple for end position.
            
        Returns:
            List of (row, col) tuples representing the optimal path, or None if unreachable.
            
        Raises:
            ValueError: If start/end are out of bounds or located on a wall.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds.")
            
        # Validate walls
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            raise ValueError("Start or end position is a wall.")

        # Handle start == end
        if start == end:
            return [start]

        # A* initialization
        open_set = []
        counter = 0  # Tie-breaker for heapq stability
        g_score = {start: 0}
        f_score = {start: self._manhattan(start, end)}
        heapq.heappush(open_set, (f_score[start], counter, start))
        came_from = {}

        # 4-directional movement
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)

                # Check bounds
                if not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols):
                    continue
                # Check wall
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue

                # Cost to enter neighbor cell
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._manhattan(neighbor, end)
                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))

        return None

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Calculate Manhattan distance between two grid points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _reconstruct_path(came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from start to current node."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


# ==================== PYTEST TESTS ====================

def test_basic_path():
    """Test straightforward path on an empty grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Optimal length = Manhattan distance + 1
    assert len(path) == 5


def test_around_walls():
    """Test pathfinding around a central wall."""
    grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert (1, 1) not in path  # Must avoid wall
    assert len(path) == 5


def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]


def test_no_path():
    """Test unreachable destination returns None."""
    grid = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None


def test_out_of_bounds():
    """Test ValueError for coordinates outside grid dimensions."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 0))


def test_weighted_optimal():
    """Test that A* correctly chooses lower-weight paths over shorter geometric paths."""
    grid = [
        [1, 10, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    assert (0, 1) not in path  # Should avoid high-cost cell
    
    # Verify optimality by calculating total traversal cost
    cost = sum(grid[r][c] for r, c in path[1:])  # Exclude start cell cost
    assert cost == 6  # Optimal route goes around the bottom (cost 1 per step)