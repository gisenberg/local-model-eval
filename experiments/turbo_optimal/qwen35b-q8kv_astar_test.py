import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D grid of movement costs.
        
        Args:
            grid: A 2D list where 0 represents a wall and positive integers 
                  represent the cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: The starting (row, col) coordinate.
            end: The ending (row, col) coordinate.
            
        Returns:
            A list of (row, col) tuples representing the path, or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # 1. Validate Bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start coordinate {start} is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End coordinate {end} is out of bounds.")
            
        # 2. Check Walls
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
            
        # 3. Check Start == End
        if start == end:
            return [start]
            
        # A* Initialization
        open_set: List[Tuple[int, Tuple[int, int]]] = []
        # Heap stores (f_score, node)
        heapq.heappush(open_set, (0, start))
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        f_score: Dict[Tuple[int, int], int] = {start: self._heuristic(start, end)}
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            # Optimization: If we found a better path to current already, skip
            if current_f > f_score.get(current, float('inf')):
                continue
                
            if current == end:
                return self._reconstruct_path(came_from, start, end)
            
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                # Check bounds
                if not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols):
                    continue
                
                # Check wall
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue
                
                # Calculate tentative g_score
                # Cost to enter neighbor cell
                neighbor_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + neighbor_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
        return None

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], 
                          start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to end using the came_from dictionary."""
        path = [end]
        current = end
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

import pytest
from typing import List, Tuple, Optional


def calculate_path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Helper to calculate total cost of a path (excluding start node cost)."""
    if not path:
        return 0
    total = 0
    for r, c in path[1:]:
        total += grid[r][c]
    return total

class TestAStarGrid:
    def test_simple_path_uniform_grid(self):
        """Test simple path on a uniform grid."""
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
        # Cost should be 4 (4 moves, cost 1 each)
        assert calculate_path_cost(grid, path) == 4

    def test_path_around_obstacles(self):
        """Test path around obstacles."""
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        # Path must go around (1,1)
        assert (1, 1) not in path

    def test_weighted_grid(self):
        """Test weighted grid prefers lower-cost cells."""
        grid = [
            [1, 10, 1],
            [1, 10, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (0, 2))
        assert path is not None
        # Should prefer going down to row 2 then right
        assert (2, 0) in path
        assert (2, 1) in path
        assert (2, 2) in path
        # Cost via bottom: 1+1+1 = 3
        # Cost via top: 10+1 = 11
        assert calculate_path_cost(grid, path) == 3

    def test_no_path_exists(self):
        """Test no path exists (fully blocked)."""
        grid = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (0, 2))
        assert path is None

    def test_start_equals_end(self):
        """Test start equals end."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (0, 0))
        assert path == [(0, 0)]

    def test_invalid_coordinates(self):
        """Test invalid coordinates raise ValueError."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (0, 0))
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (2, 2))
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (0, 5))

    def test_start_or_end_is_wall(self):
        """Test start or end is a wall returns None."""
        grid = [
            [0, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        # Start is wall
        assert astar.find_path((0, 0), (1, 1)) is None
        # End is wall
        assert astar.find_path((1, 1), (0, 0)) is None