import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    0 represents an impassable wall, positive integers represent traversal costs.
    """
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.
        
        Args:
            grid: 2D list where 0 is a wall and positive integers are cell weights.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _validate_pos(self, pos: Tuple[int, int]) -> None:
        """Validate that a position is within bounds and not a wall."""
        r, c = pos
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(f"Position {pos} is out of bounds.")
        if self.grid[r][c] == 0:
            raise ValueError(f"Position {pos} is a wall.")

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic for 4-directional movement."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: (row, col) tuple of the starting position.
            end: (row, col) tuple of the target position.
            
        Returns:
            A list of (row, col) tuples representing the optimal path,
            or None if no valid path exists.
        """
        self._validate_pos(start)
        self._validate_pos(end)

        if start == end:
            return [start]

        # Priority queue stores (f_score, counter, position)
        # Counter ensures deterministic tie-breaking and avoids tuple comparison
        counter = 0
        open_set = [(self._heuristic(start, end), counter, start)]
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # 4-directional movement: right, left, down, up
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path by backtracking through came_from
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                return path[::-1]

            for dr, dc in directions:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)

                # Skip out-of-bounds
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                # Skip walls
                if self.grid[nr][nc] == 0:
                    continue

                move_cost = self.grid[nr][nc]
                tentative_g = g_score[current] + move_cost

                # Relaxation step
                if tentative_g < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    came_from[neighbor] = current

        return None

import pytest

def test_basic_pathfinding():
    """Test standard pathfinding on an open grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 0)) == [(0, 0)]

def test_no_path_exists():
    """Test when the target is completely blocked by walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_wall_positions_raise_value_error():
    """Test that starting or ending on a wall raises ValueError."""
    grid = [[1, 0], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((0, 1), (1, 1))  # start on wall
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 1))  # end on wall

def test_weighted_optimal_path():
    """Test that A* chooses the cheaper path over the shorter path."""
    # Grid where going through (0,1) costs 10, but going around costs 4
    grid = [
        [1, 10, 1],
        [1,  0, 1],
        [1,  1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    # Optimal path must avoid the heavy cell (0, 1)
    assert (0, 1) not in path
    # Verify exact optimal route
    expected = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert path == expected