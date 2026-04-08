from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid with movement costs.
        0 represents impassable walls, positive integers represent movement costs.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not a wall."""
        row, col = pos
        return (0 <= row < self.rows and 
                0 <= col < self.cols and 
                self.grid[row][col] != 0)
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid 4-directional neighbors."""
        row, col = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        neighbors = []
        for dr, dc in directions:
            new_pos = (row + dr, col + dc)
            if self._is_valid(new_pos):
                neighbors.append(new_pos)
        return neighbors
    
    def _heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        Returns list of coordinates or None if no path exists.
        """
        # Check bounds first (raise ValueError if out of bounds)
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start position {start} is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End position {end} is out of bounds")
        
        # Check if start or end is a wall (return None)
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        
        # Handle start == end case
        if start == end:
            return [start]
        
        # A* algorithm
        # Priority queue: (f_score, g_score, position)
        open_set = [(self._heuristic(start, end), 0, start)]
        g_scores = {start: 0}  # Cost from start to each position
        came_from = {}  # For path reconstruction
        
        while open_set:
            f_score, g_score, current = heapq.heappop(open_set)
            
            # Found the goal
            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                # Cost to move to neighbor (cost of entering the neighbor cell)
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g_score = g_score + move_cost
                
                # If this is a better path to neighbor, update it
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self._heuristic(neighbor, end)
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))
        
        # No path found
        return None

import pytest

def test_simple_path_uniform_grid():
    """Test basic pathfinding on uniform cost grid."""
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
    
    # Verify path validity - consecutive cells must be adjacent
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1
    
    # Verify optimality - shortest path should have 5 cells with total cost 4
    total_cost = sum(grid[r][c] for r, c in path[1:])  # Don't count start cell
    assert total_cost == 4

def test_path_around_obstacles():
    """Test pathfinding around wall obstacles."""
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 3))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 3)
    
    # Verify no wall cells in path
    for r, c in path:
        assert grid[r][c] != 0

def test_weighted_grid():
    """Test that A* prefers lower-cost paths."""
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    
    # Verify path avoids expensive middle column
    for r, c in path:
        if c == 1:  # Middle column
            assert grid[r][c] != 10
    
    # Verify optimality - path should cost 4, not 22
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4

def test_no_path_exists():
    """Test when no valid path exists."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_start_equals_end():
    """Test when start and end are the same position."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_invalid_coordinates():
    """Test error handling for invalid coordinates."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    # Test out of bounds start
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    
    # Test out of bounds end
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))
    
    # Test wall as start
    grid_with_wall = [[0, 1], [1, 1]]
    astar_wall = AStarGrid(grid_with_wall)
    path = astar_wall.find_path((0, 0), (1, 1))
    assert path is None
    
    # Test wall as end
    path = astar_wall.find_path((1, 1), (0, 0))
    assert path is None