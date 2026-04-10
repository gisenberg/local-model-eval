import heapq
from typing import List, Tuple, Optional


class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.
    
    Grid values represent movement cost (0 = impassable wall, positive int = cost to enter cell).
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.
        
        Args:
            grid: 2D list where values represent movement cost (0 = wall, positive = cost)
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting position (row, col)
            end: Ending position (row, col)
            
        Returns:
            List of coordinates from start to end inclusive, or None if no path exists
            
        Raises:
            ValueError: If start or end is out of bounds
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position out of bounds")
            
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
            
        # If start equals end
        if start == end:
            return [start]
            
        # A* algorithm
        # Priority queue: (f_score, g_score, row, col)
        open_set = [(0, 0, start[0], start[1])]
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: dict[Tuple[int, int], int] = {start: 0}
        closed_set: set[Tuple[int, int]] = set()
        
        while open_set:
            f, g, current_row, current_col = heapq.heappop(open_set)
            current = (current_row, current_col)
            
            # Skip if already processed
            if current in closed_set:
                continue
                
            # Check if reached goal
            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            closed_set.add(current)
            
            # Explore 4-directional neighbors (up, down, left, right)
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current_row + dr, current_col + dc)
                
                # Check bounds
                if not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols):
                    continue
                    
                # Check if wall
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue
                    
                # Calculate tentative g_score (cost to reach neighbor)
                tentative_g = g + self.grid[neighbor[0]][neighbor[1]]
                
                # Skip if already closed
                if neighbor in closed_set:
                    continue
                    
                # Update if better path found
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    h = self._manhattan_distance(neighbor, end)
                    f = tentative_g + h
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (f, tentative_g, neighbor[0], neighbor[1]))
                    
        return None
        
    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Test suite
import pytest


def test_simple_path_uniform_grid():
    """Test simple path on uniform grid with cost 1."""
    grid = [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    
    # Verify path validity (consecutive cells are adjacent)
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1
    
    # Calculate cost (excluding start)
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4  # Manhattan distance from (0,0) to (2,2) is 4


def test_path_around_obstacles():
    """Test path finding around obstacles."""
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    
    # Verify no wall cells in path
    for r, c in path:
        assert grid[r][c] != 0
        
    # Verify path validity
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1
    
    # Calculate cost - must go around obstacles (length 6)
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 6


def test_weighted_grid():
    """Test that path prefers lower-cost cells."""
    grid = [
        [1, 10, 10, 1],
        [1, 10, 10, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 3))
    
    assert path is not None
    
    # Verify path validity
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1
    
    # Calculate cost
    total_cost = sum(grid[r][c] for r, c in path[1:])
    
    # Optimal path should go down the left side (cost 1 each) then across bottom
    # (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(2,3) = 5 steps, cost = 5
    # Alternative via top: cost = 10+10+1+1+1 = 23
    assert total_cost == 5


def test_no_path_exists():
    """Test when no path exists due to obstacles."""
    grid = [
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None


def test_start_equals_end():
    """Test when start equals end."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]


def test_invalid_coordinates():
    """Test invalid coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    # Negative indices
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, -1), (0, 0))
    
    # Out of bounds
    with pytest.raises(ValueError):
        astar.find_path((2, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 2), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (0, 2))