import heapq
from typing import List, Optional, Tuple, Set, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.

    Attributes:
        grid (List[List[int]]): 2D grid where 0 represents walls and positive integers represent weights.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D grid.

        Args:
            grid: 2D list where 0 represents walls and positive integers represent weights.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance heuristic between two positions.

        Args:
            pos: Current position (row, col).
            end: Target position (row, col).

        Returns:
            Manhattan distance between pos and end.
        """
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    def _is_valid(self, row: int, col: int) -> bool:
        """
        Check if a position is within grid bounds and not a wall.

        Args:
            row: Row index.
            col: Column index.

        Returns:
            True if position is valid, False otherwise.
        """
        return (0 <= row < self.rows and
                0 <= col < self.cols and
                self.grid[row][col] != 0)

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.

        Args:
            start: Starting position (row, col).
            end: Target position (row, col).

        Returns:
            List of positions representing the path from start to end, or None if no path exists.

        Raises:
            ValueError: If start or end positions are out of bounds or on walls.
        """
        # Validate start and end positions
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds")
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is on a wall")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is on a wall")

        # Handle case where start equals end
        if start == end:
            return [start]

        # Initialize data structures
        open_heap: List[Tuple[int, int, int, Tuple[int, int]]] = []
        heapq.heappush(open_heap, (0, 0, 0, start))

        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        closed_set: Set[Tuple[int, int]] = set()

        # Directions: up, right, down, left
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

        while open_heap:
            f_score, g_val, _, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue

            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]

            closed_set.add(current)

            # Explore neighbors
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)

                if not self._is_valid(neighbor[0], neighbor[1]):
                    continue

                if neighbor in closed_set:
                    continue

                # Calculate tentative g score
                weight = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_val + weight

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f_score, tentative_g, 0, neighbor))

        return None  # No path found


# Tests
def test_basic_path():
    """Test finding a simple path in a small grid."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Verify it's a valid path
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1  # Adjacent cells


def test_start_equals_end():
    """Test when start and end are the same."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]


def test_walls_block_path():
    """Test that walls properly block paths."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    assert len(path) == 5  # Must go around the wall


def test_no_path_exists():
    """Test when no path exists due to complete wall blocking."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None


def test_out_of_bounds_raises_error():
    """Test that out of bounds positions raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)

    try:
        astar.find_path((-1, 0), (1, 1))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    try:
        astar.find_path((0, 0), (2, 0))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_weighted_optimal_path():
    """Test that A* finds optimal path considering weights."""
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    # Optimal path should go down, right twice, then up
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]