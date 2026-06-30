import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Uses 4-directional movement, Manhattan heuristic, and a priority queue (heapq).
    Grid cells with value 0 represent walls (impassable). Positive values represent 
    the cost to enter/traverse that cell.
    """
    def __init__(self, grid: List[List[float]]):
        """
        Initialize the grid for pathfinding.
        
        :param grid: 2D list of weights. 0 = wall, >0 = traversal cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within grid boundaries."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _heuristic(self, r: int, c: int, end: Tuple[int, int]) -> float:
        """Manhattan distance heuristic (admissible and consistent for 4-directional grids)."""
        return abs(r - end[0]) + abs(c - end[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* search.
        
        :param start: (row, col) tuple for starting position.
        :param end: (row, col) tuple for target position.
        :return: List of (row, col) tuples representing the optimal path, or None if unreachable.
        :raises ValueError: If coordinates are out of bounds or start/end positions are walls.
        """
        # Validate coordinates
        if not self._is_valid(*start) or not self._is_valid(*end):
            raise ValueError("Start or end coordinates are out of bounds.")
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            raise ValueError("Start or end position is a wall.")

        # Trivial case
        if start == end:
            return [start]

        # Priority queue: (f_score, tie_breaker_counter, (row, col))
        open_set = []
        counter = 0
        h_start = self._heuristic(*start, end)
        heapq.heappush(open_set, (h_start, counter, start))
        counter += 1

        g_score = {start: 0.0}
        came_from = {}
        closed_set = set()

        # 4-directional movement: right, left, down, up
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_set:
            _, _, current = heapq.heappop(open_set)

            # Goal reached
            if current == end:
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                return path[::-1]

            # Skip already expanded nodes
            if current in closed_set:
                continue
            closed_set.add(current)

            # Explore neighbors
            for dr, dc in directions:
                nr, nc = current[0] + dr, current[1] + dc
                
                if not self._is_valid(nr, nc):
                    continue
                if self.grid[nr][nc] == 0:  # Wall
                    continue
                if (nr, nc) in closed_set:
                    continue

                tentative_g = g_score[current] + self.grid[nr][nc]
                
                # Found a better path to this neighbor
                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + self._heuristic(nr, nc, end)
                    heapq.heappush(open_set, (f_score, counter, (nr, nc)))
                    counter += 1

        # Exhausted search space without reaching end
        return None

import pytest
from typing import List, Tuple, Optional

# Import the class (adjust path if in separate file)

def test_basic_pathfinding():
    """Test standard pathfinding on a simple grid with a wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]

def test_start_equals_end():
    """Test trivial case where start and end are the same."""
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_no_path_exists():
    """Test that None is returned when end is unreachable."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))

def test_wall_at_start_or_end_raises_value_error():
    """Test that starting or ending on a wall raises ValueError."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))
        
    grid2 = [[1, 0], [1, 1]]
    astar2 = AStarGrid(grid2)
    with pytest.raises(ValueError, match="wall"):
        astar2.find_path((0, 0), (0, 1))

def test_weighted_grid_optimal_path():
    """Test that A* chooses lowest-cost path over shortest-step path."""
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Direct path cost: 10 + 1 = 11
    # Bottom path cost: 1+1+1+1+1+1 = 6 (optimal)
    path = astar.find_path((0, 0), (0, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]