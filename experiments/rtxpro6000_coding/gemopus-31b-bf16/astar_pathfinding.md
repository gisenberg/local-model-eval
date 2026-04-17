```python
import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* Pathfinding implementation on a weighted 2D grid.
    Grid values: 0 = wall, positive integer = cost to enter that cell.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D list of costs.
        
        Args:
            grid: A 2D list where grid[r][c] is the cost to enter cell (r, c).
                  0 indicates an impassable wall.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: (row, col) starting coordinates
            end: (row, col) destination coordinates
            
        Returns:
            List of (row, col) tuples representing the path including start and end,
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate bounds
        if not self._is_valid(*start) or not self._is_valid(*end):
            raise ValueError("Start or end coordinates are out of bounds")

        # Edge case: start is end
        if start == end:
            return [start]

        # Edge case: start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Priority queue: (f_score, (row, col))
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # Track where we came from to reconstruct path
        came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        
        # Cost from start to current node
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        
        # Estimated total cost (g_score + heuristic)
        f_score: Dict[Tuple[int, int], float] = {start: self._heuristic(start, end)}

        while open_set:
            # Get node with lowest f_score
            _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from.get(current)
                return path[::-1]

            r, c = current
            # 4-directional movement
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (r + dr, c + dc)
                
                if not self._is_valid(*neighbor):
                    continue
                
                weight = self.grid[neighbor[0]][neighbor[1]]
                if weight == 0:  # Wall
                    continue
                
                # Tentative g_score is distance from start to neighbor through current
                tentative_g_score = g_score[current] + weight
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    # This path to neighbor is better than any previous one
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

def calculate_path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Helper to calculate total cost of a path (excluding start cell cost)."""
    if not path or len(path) < 2:
        return 0
    return sum(grid[r][c] for r, c in path[1:])

# --- Pytest Tests ---

import pytest

def test_simple_path():
    """Test pathfinding on a uniform grid with no obstacles."""
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 2)
    path = astar.find_path(start, end)
    
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    assert len(path) == 5  # Manhattan distance + 1
    assert calculate_path_cost(grid, path) == 4

def test_path_around_obstacles():
    """Test pathfinding around walls (0s)."""
    grid = [[1, 1, 1],
            [0, 0, 1],
            [1, 1, 1]]
    astar = AStarGrid(grid)
    start, end = (0, 0), (2, 0)
    path = astar.find_path(start, end)
    
    # Must go around the wall: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (2,1) -> (2,0)
    assert path is not None
    assert len(path) == 7
    assert (1, 0) not in path  # Should not pass through wall
    assert calculate_path_cost(grid, path) == 6

def test_weighted_grid():
    """Test that A* prefers lower-cost cells over shorter distance."""
    # Path A (direct): (0,0) -> (0,1) -> (0,2) | Cost: 10 + 10 = 20
    # Path B (detour): (0,0) -> (1,0) -> (1,1) -> (1,2) -> (0,2) | Cost: 1+1+1+1 = 4
    grid = [[1, 10, 1],
            [1, 1, 1]]
    astar = AStarGrid(grid)
    start, end = (0, 0), (0, 2)
    path = astar.find_path(start, end)
    
    assert path == [(0, 0), (1, 0), (1, 1), (1, 2), (0, 2)]
    assert calculate_path_cost(grid, path) == 4

def test_no_path_exists():
    """Test when start and end are separated by walls."""
    grid = [[1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]]
    astar = AStarGrid(grid)
    assert astar.find_path((0, 0), (2, 2)) is None

def test_start_equals_end():
    """Test when start and end are the same coordinate."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    start = (0, 0)
    assert astar.find_path(start, start) == [start]

def test_invalid_coordinates():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 0))
```