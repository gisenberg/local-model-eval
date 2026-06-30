import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid representation:
    - 0: Wall (impassable)
    - >0: Traversable cell with movement cost equal to its value
    
    Movement is 4-directional (up, down, left, right).
    Uses Manhattan distance as heuristic and heapq for the priority queue.
    """
    
    def __init__(self, grid: List[List[float]]):
        """
        Initialize the grid.
        
        Args:
            grid: 2D list where 0 represents walls and positive numbers represent weights.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty")
        
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A*.
        
        Args:
            start: (row, col) tuple of the starting position.
            end: (row, col) tuple of the target position.
            
        Returns:
            List of (row, col) tuples representing the optimal path, or None if unreachable.
            
        Raises:
            ValueError: If start or end coordinates are out of grid bounds.
        """
        # 1. Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds")

        # 2. Handle start == end
        if start == end:
            return [start]

        # 3. Handle walls at start/end
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # A* initialization
        # Priority queue stores: (f_score, tie_breaker_counter, (row, col))
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))
        
        g_score = {start: 0}
        came_from = {}
        closed_set = set()
        counter = 1  # Ensures stable sorting in heapq when f_scores tie

        while open_set:
            f, _, current = heapq.heappop(open_set)

            # Skip if already processed optimally
            if current in closed_set:
                continue

            # Goal reached
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            closed_set.add(current)

            # Explore neighbors
            for dr, dc in self.directions:
                nr, nc = current[0] + dr, current[1] + dc

                # Boundary check
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                
                # Wall check
                if self.grid[nr][nc] == 0:
                    continue
                
                # Skip if already closed
                if (nr, nc) in closed_set:
                    continue

                # Calculate tentative g-score
                tentative_g = g_score[current] + self.grid[nr][nc]

                # Update if this path is better
                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + self._heuristic((nr, nc), end)
                    heapq.heappush(open_set, (f_score, counter, (nr, nc)))
                    counter += 1

        # No path found
        return None

import pytest

def test_basic_open_grid_path():
    """Test standard pathfinding on an unweighted open grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

def test_start_equals_end():
    """Test immediate return when start and end are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_wall_blocks_path():
    """Test that completely blocked paths return None."""
    grid = [[1, 0, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_start_or_end_is_wall():
    """Test that starting or ending on a wall returns None."""
    grid = [[0, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (0, 2)) is None
    assert astar.find_path((0, 0), (0, 0)) is None  # Wall at start/end

def test_out_of_bounds_raises_value_error():
    """Test that invalid coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (0, 0))
        
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_weighted_optimal_path():
    """Test that A* chooses lower-cost path over shorter Manhattan distance."""
    # Direct path cost: 100 + 1 = 101
    # Around path cost: 1 + 1 + 1 + 1 + 1 + 1 = 6
    grid = [
        [1, 100, 1],
        [1,   0, 1],
        [1,   1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is not None
    expected = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]
    assert path == expected