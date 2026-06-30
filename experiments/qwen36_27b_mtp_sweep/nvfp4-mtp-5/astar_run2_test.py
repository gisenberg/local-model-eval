import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    - 0 represents a wall (impassable)
    - Positive values represent traversal costs to enter a cell
    - Uses 4-directional movement and Manhattan distance heuristic
    - Guarantees optimal path for non-negative weights
    """
    
    def __init__(self, grid: List[List[float]]):
        """
        Initialize the grid.
        
        :param grid: 2D list of numeric weights. 0 = wall, >0 = traversable cost.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic (admissible and consistent for 4-directional grids)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end.
        
        :param start: (row, col) tuple
        :param end: (row, col) tuple
        :return: List of (row, col) tuples representing the path, or None if unreachable.
        :raises ValueError: If coordinates are out of bounds or land on walls.
        """
        # 1. Validate coordinates
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start coordinates are out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End coordinates are out of bounds.")
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall.")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall.")

        # 2. Handle immediate success
        if start == end:
            return [start]

        # 3. Initialize A* structures
        # Priority queue stores (f_score, tie_breaker, position)
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        initial_f = self._heuristic(start, end)
        heapq.heappush(open_set, (initial_f, 0, start))
        
        g_score: dict[Tuple[int, int], float] = {start: 0}
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        counter = 1  # Tie-breaker for heapq stability

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # 4. Main loop
        while open_set:
            f, _, current = heapq.heappop(open_set)
            
            # Skip stale entries (lazy deletion optimization)
            if f > g_score.get(current, float('inf')):
                continue

            # Goal reached
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Explore neighbors
            for dr, dc in directions:
                nr, nc = current[0] + dr, current[1] + dc
                
                # Bounds and wall check
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != 0:
                    tentative_g = g_score[current] + self.grid[nr][nc]
                    
                    if tentative_g < g_score.get((nr, nc), float('inf')):
                        g_score[(nr, nc)] = tentative_g
                        f_score = tentative_g + self._heuristic((nr, nc), end)
                        heapq.heappush(open_set, (f_score, counter, (nr, nc)))
                        came_from[(nr, nc)] = current
                        counter += 1

        # 5. No path found
        return None

import pytest
from typing import List, Tuple, Optional

# Import the class from above
# 
def test_basic_path():
    """Test straightforward pathfinding on an open grid."""
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
    assert len(path) == 5  # 4 moves + start

def test_start_equals_end():
    """Test immediate success when start and end are identical."""
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_unreachable_path():
    """Test that surrounded start/end returns None."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_out_of_bounds():
    """Test ValueError for out-of-bounds coordinates."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_wall_at_start_or_end():
    """Test ValueError when start or end lands on a wall."""
    grid = [
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 0]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((1, 1), (2, 2))

def test_weighted_optimality():
    """Test that A* chooses the lowest-cost path over the shortest Manhattan path."""
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Optimal route avoids the 10s by going down then right
    expected = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert path == expected