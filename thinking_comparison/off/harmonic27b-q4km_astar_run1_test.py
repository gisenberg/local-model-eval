from typing import List, Tuple, Optional
import heapq
import math

class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.
    
    Grid values represent movement cost:
    - 0: impassable wall
    - positive int: cost to enter that cell
    
    Supports 4-directional movement (up, down, left, right).
    Uses Manhattan distance as heuristic.
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid with the given 2D grid.
        
        Args:
            grid: 2D list where each value represents movement cost
                  (0 = wall, positive int = cost to enter cell)
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], int]]:
        """
        Get valid neighboring positions and their movement costs.
        
        Returns list of (neighbor_position, cost_to_enter) tuples.
        """
        row, col = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        neighbors = []
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            new_pos = (new_row, new_col)
            
            if self._is_valid(new_pos):
                cost = self.grid[new_row][new_col]
                if cost > 0:  # Only add if not a wall
                    neighbors.append((new_pos, cost))
        
        return neighbors
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting position as (row, col) tuple
            end: Ending position as (row, col) tuple
            
        Returns:
            List of (row, col) coordinates representing the path from start to end,
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds
        """
        # Validate coordinates
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end coordinates are out of bounds")
        
        # Handle edge case: start equals end
        if start == end:
            return [start]
        
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        
        # A* algorithm initialization
        # Priority queue: (f_score, g_score, current_pos)
        open_set = [(0, 0, start)]
        
        # Track g_scores (cost from start to each position)
        g_scores = {start: 0}
        
        # Track parent positions for path reconstruction
        parents = {}
        
        # Set of positions in open_set for efficient lookup
        open_set_positions = {start}
        
        while open_set:
            # Get position with lowest f_score
            _, current_g, current_pos = heapq.heappop(open_set)
            
            # Remove from open_set
            open_set_positions.remove(current_pos)
            
            # Check if we reached the goal
            if current_pos == end:
                # Reconstruct path
                path = []
                current = end
                while current in parents:
                    path.append(current)
                    current = parents[current]
                path.append(start)
                return path[::-1]  # Reverse to get start to end
            
            # Explore neighbors
            for neighbor, move_cost in self._get_neighbors(current_pos):
                tentative_g = current_g + move_cost
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    parents[neighbor] = current_pos
                    
                    # Calculate f_score = g_score + heuristic
                    f_score = tentative_g + self._manhattan_distance(neighbor, end)
                    
                    # Add to open_set if not already there
                    if neighbor not in open_set_positions:
                        heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                        open_set_positions.add(neighbor)
        
        # No path found
        return None

import pytest
from typing import List, Tuple, Optional

# Import the AStarGrid class from the previous code
  # Replace with actual import

def test_simple_path_uniform_grid():
    """Test simple path on uniform grid with cost 1."""
    grid = [
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (3, 3))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (3, 3)
    assert len(path) == 7  # Minimum path length
    
    # Verify path validity
    for i in range(len(path) - 1):
        curr, next_pos = path[i], path[i+1]
        assert abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1]) == 1
    
    # Verify optimality (total cost should be 6)
    total_cost = sum(grid[r][c] for r, c in path[1:])  # Don't count start
    assert total_cost == 6

def test_path_around_obstacles():
    """Test path finding around obstacles."""
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (3, 3))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (3, 3)
    
    # Verify path doesn't go through walls
    for r, c in path:
        assert grid[r][c] != 0

def test_weighted_grid_optimality():
    """Test that algorithm prefers lower-cost paths."""
    grid = [
        [1, 10, 1],
        [1, 1, 1],
        [1, 10, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    
    # The optimal path should go through the middle row (cost 1) rather than top/bottom (cost 10)
    # Expected path: (0,0) -> (1,0) -> (1,1) -> (1,2) -> (2,2)
    expected_cost = 1 + 1 + 1 + 1  # 4 steps of cost 1 each
    actual_cost = sum(grid[r][c] for r, c in path[1:])
    assert actual_cost == expected_cost

def test_no_path_exists():
    """Test when no path exists due to complete blockage."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is None

def test_start_equals_end():
    """Test when start position equals end position."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    
    assert path == [(1, 1)]

def test_invalid_coordinates():
    """Test that invalid coordinates raise ValueError."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Test out of bounds start
    with pytest.raises(ValueError):
        astar.find_path((5, 5), (1, 1))
    
    # Test out of bounds end
    with pytest.raises(ValueError):
        astar.find_path((1, 1), (5, 5))
    
    # Test negative coordinates
    with pytest.raises(ValueError):
        astar.find_path((-1, 1), (1, 1))