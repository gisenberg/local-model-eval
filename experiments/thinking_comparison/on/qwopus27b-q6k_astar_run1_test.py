import heapq
from typing import List, Tuple, Optional


class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.
    
    Grid values represent movement cost: 0 = impassable wall, 
    positive int = cost to enter that cell.
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D grid of costs.
        
        Args:
            grid: 2D list where grid[r][c] is the cost to enter cell (r, c).
                  0 represents an impassable wall.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting position as (row, col) tuple.
            end: Ending position as (row, col) tuple.
            
        Returns:
            List of (row, col) tuples representing the path from start to end inclusive,
            or None if no path exists.
            
        Raises:
            ValueError: If start or end is out of bounds.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start position {start} is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End position {end} is out of bounds")
        
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        
        # If start equals end, return single-element path
        if start == end:
            return [start]
        
        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Priority queue: (f_score, g_score, current_node)
        # g_score is used as tie-breaker to ensure consistent ordering
        open_set = [(self._manhattan(start, end), 0, start)]
        
        # Track best known cost to reach each node
        g_score = {start: 0}
        
        # Track parent for path reconstruction
        came_from = {}
        
        while open_set:
            f, g, current = heapq.heappop(open_set)
            
            # Skip if we've found a better path to this node already
            if g > g_score[current]:
                continue
            
            # Check if we reached the goal
            if current == end:
                return self._reconstruct_path(came_from, current)
            
            # Explore neighbors
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                # Check bounds
                if not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols):
                    continue
                
                # Check if wall
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue
                
                # Calculate cost to reach neighbor (cost to enter neighbor cell)
                tentative_g = g + self.grid[neighbor[0]][neighbor[1]]
                
                # If this is a better path to neighbor, update and add to open set
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._manhattan(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        # No path found
        return None
    
    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from start to current by following came_from pointers."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


# ==================== PYTEST TESTS ====================

import pytest


def test_simple_path_uniform_grid():
    """Test simple path on uniform grid (all costs 1)."""
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
    
    # Verify path validity (consecutive cells are adjacent)
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1
    
    # Calculate cost (excluding start cell)
    cost = sum(grid[r][c] for r, c in path[1:])
    # Optimal path length is 4 steps (Manhattan distance), cost = 4
    assert cost == 4


def test_path_around_obstacles():
    """Test path finding around obstacles (walls)."""
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
    
    # Verify no walls in path
    for r, c in path:
        assert grid[r][c] != 0
    
    # Verify path validity
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1


def test_weighted_grid():
    """Test that algorithm prefers lower-cost cells over shorter paths."""
    # Grid where middle path is expensive (100) but shorter in steps
    # Edge path is cheaper (1) but longer in steps
    grid = [
        [1, 1, 1, 1],
        [1, 100, 100, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 3))
    
    assert path is not None
    
    # Calculate cost
    cost = sum(grid[r][c] for r, c in path[1:])
    
    # Optimal path should go around the edges, avoiding the 100-cost cells
    # Cost should be 5 (5 steps of cost 1 each)
    assert cost == 5
    
    # Verify no high-cost cells in path
    for r, c in path:
        assert grid[r][c] != 100


def test_no_path_exists():
    """Test that None is returned when no path exists."""
    # Case 1: Fully blocked
    grid1 = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar1 = AStarGrid(grid1)
    assert astar1.find_path((0, 0), (0, 2)) is None
    
    # Case 2: Start is wall
    grid2 = [
        [0, 1],
        [1, 1]
    ]
    astar2 = AStarGrid(grid2)
    assert astar2.find_path((0, 0), (1, 1)) is None
    
    # Case 3: End is wall
    assert astar2.find_path((0, 1), (0, 0)) is None


def test_start_equals_end():
    """Test that start == end returns [start]."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]


def test_invalid_coordinates():
    """Test that ValueError is raised for out-of-bounds coordinates."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    # Test negative coordinates
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    
    with pytest.raises(ValueError):
        astar.find_path((0, -1), (1, 1))
    
    # Test coordinates beyond grid size
    with pytest.raises(ValueError):
        astar.find_path((2, 0), (1, 1))
    
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))