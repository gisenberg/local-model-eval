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
        Initialize the AStarGrid with a 2D grid.
        
        Args:
            grid: 2D list where values represent movement cost.
                  0 = impassable wall, positive int = cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting coordinates (row, col)
            end: Ending coordinates (row, col)
            
        Returns:
            List of coordinates from start to end inclusive, or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate coordinates
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start coordinates out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End coordinates out of bounds")
        
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        
        # If start equals end
        if start == end:
            return [start]
        
        # A* algorithm
        # Priority queue: (f_score, g_score, row, col)
        # g_score is used as tiebreaker to ensure consistent ordering
        open_set = [(self._heuristic(start, end), 0, start[0], start[1])]
        
        # Track g_scores (cost from start to node)
        g_scores = {start: 0}
        
        # Track came_from for path reconstruction
        came_from = {}
        
        while open_set:
            f_score, g_score, current_row, current_col = heapq.heappop(open_set)
            current = (current_row, current_col)
            
            # Skip if we've found a better path to this node already
            if current in g_scores and g_scores[current] < g_score:
                continue
            
            # If we reached the goal
            if current == end:
                return self._reconstruct_path(came_from, current)
            
            # Explore neighbors
            for dr, dc in self.directions:
                neighbor_row, neighbor_col = current_row + dr, current_col + dc
                neighbor = (neighbor_row, neighbor_col)
                
                # Check bounds
                if not (0 <= neighbor_row < self.rows and 0 <= neighbor_col < self.cols):
                    continue
                
                # Check if wall
                if self.grid[neighbor_row][neighbor_col] == 0:
                    continue
                
                # Calculate cost to move to neighbor (cost to enter neighbor cell)
                move_cost = self.grid[neighbor_row][neighbor_col]
                tentative_g_score = g_score + move_cost
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor_row, neighbor_col))
                    came_from[neighbor] = current
        
        return None
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two points.
        
        Args:
            a: First point (row, col)
            b: Second point (row, col)
            
        Returns:
            Manhattan distance between a and b.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct path from came_from dictionary.
        
        Args:
            came_from: Dictionary mapping each node to its predecessor
            current: Current node (should be the goal)
            
        Returns:
            List of coordinates from start to current.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]  # Reverse to get start to end


# Test suite
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
    assert len(path) == 5  # 5 nodes including start and end
    
    # Verify path validity (each step is adjacent)
    for i in range(1, len(path)):
        r1, c1 = path[i-1]
        r2, c2 = path[i]
        assert abs(r1 - r2) + abs(c1 - c2) == 1, "Path contains non-adjacent steps"
    
    # Calculate total cost (sum of costs to enter each cell except start)
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4  # 4 moves, each cost 1

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
    
    # Verify no wall is in path
    for r, c in path:
        assert grid[r][c] != 0, f"Path contains wall at ({r}, {c})"
    
    # Verify optimality: should be 5 steps (cost 5)
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 5

def test_weighted_grid():
    """Test that path prefers lower-cost cells."""
    # Create a grid where going through the middle is expensive
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
    
    # Verify path avoids expensive cells (0,1) and (1,1)
    for r, c in path:
        assert not (r == 0 and c == 1), "Path should avoid expensive cell (0,1)"
        assert not (r == 1 and c == 1), "Path should avoid expensive cell (1,1)"
    
    # Calculate cost - should be 4 (going around the bottom)
    # (0,0)->(1,0)->(2,0)->(2,1)->(2,2) costs 1+1+1+1 = 4
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4

def test_no_path_exists():
    """Test when no path exists due to walls."""
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
    """Test invalid coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    # Test start out of bounds
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    
    # Test end out of bounds
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 0))
    
    # Test start is wall
    grid_with_wall = [[0, 1], [1, 1]]
    astar_wall = AStarGrid(grid_with_wall)
    path = astar_wall.find_path((0, 0), (0, 1))
    assert path is None
    
    # Test end is wall
    path = astar_wall.find_path((0, 1), (0, 0))
    assert path is None