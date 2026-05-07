import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid conventions:
    - 0 represents a wall (impassable)
    - Positive integers represent the traversal cost to enter that cell
    - Movement is 4-directional (up, down, left, right)
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid for pathfinding.
        
        Args:
            grid: 2D list of integers representing the map.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within bounds and not a wall."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: Tuple (row, col) representing the starting position.
            end: Tuple (row, col) representing the target position.
            
        Returns:
            A list of tuples representing the optimal path from start to end,
            or None if no valid path exists.
            
        Raises:
            ValueError: If start or end is out of bounds or located on a wall.
        """
        # Validate start position
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds")
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall")
            
        # Validate end position
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall")

        # Handle trivial case
        if start == end:
            return [start]

        # Priority queue: (f_score, counter, position)
        # Counter breaks ties and ensures FIFO behavior for equal f-scores
        counter = 0
        open_set = [(0, counter, start)]
        
        # g_score: lowest known cost from start to each node
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        # came_from: tracks parent nodes for path reconstruction
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while open_set:
            f, _, current = heapq.heappop(open_set)

            # Skip stale entries (we found a better path to this node already)
            if current in g_score and f > g_score[current]:
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
                neighbor = (current[0] + dr, current[1] + dc)
                if not self._is_valid(neighbor):
                    continue

                # Cost to move is the weight of the destination cell
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))
                    came_from[neighbor] = current

        # Queue exhausted without reaching goal
        return None

import pytest

def test_basic_pathfinding():
    """Test standard pathfinding on an open grid."""
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
    assert len(path) == 5  # Manhattan distance + 1

def test_start_equals_end():
    """Test trivial case where start and end are the same."""
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_no_path_due_to_walls():
    """Test that blocked paths return None."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds_start_raises_value_error():
    """Test that out-of-bounds start position raises ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))

def test_out_of_bounds_end_raises_value_error():
    """Test that out-of-bounds end position raises ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_weighted_optimal_path():
    """Test that A* chooses lower-cost weighted path over shorter Manhattan path."""
    # Direct path cost: 10 + 1 = 11
    # Detour path cost: 1 + 1 + 1 + 1 = 4
    grid = [
        [1, 10, 1],
        [1,  1, 1],
        [1,  1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    # Optimal path should avoid the high-cost cell (0, 1)
    assert (0, 1) not in path
    assert len(path) == 5  # (0,0)->(1,0)->(1,1)->(1,2)->(0,2)