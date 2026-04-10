import heapq
from typing import List, Tuple, Optional


class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.
    
    Grid values represent movement cost: 0 = impassable wall, positive int = cost to enter cell.
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.
        
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
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position out of bounds")
        
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        
        # Edge case: start equals end
        if start == end:
            return [start]
        
        # A* algorithm
        # Priority queue stores (f_score, g_score, (row, col))
        # f_score = g_score + h_score (estimated total cost)
        open_set: List[Tuple[int, int, Tuple[int, int]]] = [(0, 0, start)]
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: dict[Tuple[int, int], int] = {start: 0}
        
        while open_set:
            f, g, current = heapq.heappop(open_set)
            
            if current == end:
                # Reconstruct path
                path: List[Tuple[int, int]] = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            # Explore 4-directional neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                # Check bounds
                if not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols):
                    continue
                
                # Check if wall
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue
                
                # Calculate tentative g_score (cost to reach neighbor)
                tentative_g = g + self.grid[neighbor[0]][neighbor[1]]
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = self._manhattan(neighbor, end)
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
        
        return None
    
    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Pytest tests
import pytest


def test_simple_path_uniform_grid():
    """Test simple path on uniform grid with cost 1."""
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
    
    # Verify optimality: cost should be 4 (entering 4 cells)
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4


def test_path_around_obstacles():
    """Test path finding around obstacles."""
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
    
    # Verify optimality: should go around the wall block
    # Minimum cost path: (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3) or similar
    # Cost = 5 (entering 5 cells)
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 5


def test_weighted_grid():
    """Test that path prefers lower-cost cells."""
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    
    # Verify path avoids expensive middle column (cost 10)
    # Optimal path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2) with cost 4
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4
    
    # Verify no expensive cells in path
    for r, c in path:
        assert grid[r][c] != 10


def test_no_path_exists():
    """Test when no path exists due to blocking walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None


def test_start_equals_end():
    """Test when start equals end."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]


def test_invalid_coordinates():
    """Test that invalid coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    # Negative coordinates
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    
    with pytest.raises(ValueError):
        astar.find_path((0, -1), (0, 0))
    
    # Out of bounds
    with pytest.raises(ValueError):
        astar.find_path((2, 0), (0, 0))
    
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 2))
    
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))