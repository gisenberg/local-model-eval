"""
A* Pathfinding Implementation on Weighted 2D Grid

This module provides an A* pathfinding algorithm implementation for finding
optimal paths on weighted 2D grids with support for obstacles and varying
movement costs.
"""

import heapq
from typing import List, Tuple, Optional


class AStarGrid:
    """
    A* pathfinding implementation for weighted 2D grids.
    
    Attributes:
        grid: 2D list representing the grid where each cell contains the
              cost to enter that cell (0 = impassable wall)
        rows: Number of rows in the grid
        cols: Number of columns in the grid
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the AStarGrid with a 2D grid.
        
        Args:
            grid: 2D list where grid[row][col] represents the cost to enter
                  that cell. 0 indicates an impassable wall, positive integers
                  represent the movement cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
    
    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """
        Check if a position is within grid bounds.
        
        Args:
            pos: Tuple of (row, col) coordinates
            
        Returns:
            True if position is within bounds, False otherwise
        """
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], int]]:
        """
        Get valid neighboring cells with their movement costs.
        
        Args:
            pos: Current position as (row, col) tuple
            
        Returns:
            List of tuples containing (neighbor_position, cost_to_enter)
        """
        row, col = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        neighbors = []
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self._is_valid((new_row, new_col)):
                cost = self.grid[new_row][new_col]
                if cost > 0:  # Not a wall
                    neighbors.append(((new_row, new_col), cost))
        
        return neighbors
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance heuristic.
        
        Args:
            pos: Current position as (row, col) tuple
            goal: Goal position as (row, col) tuple
            
        Returns:
            Manhattan distance between position and goal
        """
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: Starting position as (row, col) tuple
            end: Ending position as (row, col) tuple
            
        Returns:
            List of (row, col) tuples representing the path from start to end,
            or None if no path exists
            
        Raises:
            ValueError: If start or end coordinates are out of bounds
        """
        # Validate coordinates are within bounds
        if not self._is_valid(start):
            raise ValueError(f"Start position {start} is out of bounds")
        if not self._is_valid(end):
            raise ValueError(f"End position {end} is out of bounds")
        
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None
        
        # Edge case: start equals end
        if start == end:
            return [start]
        
        # Open set: min-heap of (f_score, g_score, position)
        # Using g_score as secondary sort key for stability
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))
        
        # Track g_score (cost from start) for each position
        g_score: dict[Tuple[int, int], int] = {start: 0}
        
        # Track parent for path reconstruction
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # Closed set: positions already evaluated
        closed_set: set[Tuple[int, int]] = set()
        
        while open_set:
            # Get position with lowest f_score
            _, current_g, current = heapq.heappop(open_set)
            
            # Skip if already processed
            if current in closed_set:
                continue
            
            # Check if we reached the goal
            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            closed_set.add(current)
            
            # Explore neighbors
            for neighbor, move_cost in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = current_g + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        # No path found
        return None


# Test module
if __name__ == "__main__":
    import pytest
    import sys
    
    # Run tests
    sys.exit(pytest.main([__file__, "-v"]))

"""
Pytest test suite for A* pathfinding implementation.
"""

import pytest
from typing import List, Tuple, Optional
import sys
import os

# Add parent directory to path for importing AStarGrid
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))




def calculate_path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Calculate total cost of a path (sum of costs to enter each cell except start)."""
    if not path:
        return 0
    total_cost = 0
    for i in range(1, len(path)):
        row, col = path[i]
        total_cost += grid[row][col]
    return total_cost


def verify_path_validity(grid: List[List[int]], path: List[Tuple[int, int]], 
                         start: Tuple[int, int], end: Tuple[int, int]) -> bool:
    """Verify that a path is valid (connected, within bounds, no walls)."""
    if not path:
        return False
    
    if path[0] != start or path[-1] != end:
        return False
    
    for i in range(len(path) - 1):
        curr_row, curr_col = path[i]
        next_row, next_col = path[i + 1]
        
        # Check adjacency (4-directional)
        row_diff = abs(curr_row - next_row)
        col_diff = abs(curr_col - next_col)
        if not ((row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)):
            return False
        
        # Check not a wall
        if grid[next_row][next_col] == 0:
            return False
    
    return True


class TestAStarGrid:
    """Test suite for AStarGrid class."""
    
    def test_simple_path_uniform_grid(self):
        """Test simple path on a uniform cost grid."""
        grid = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]
        
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (3, 3)
        
        path = astar.find_path(start, end)
        
        assert path is not None, "Path should exist"
        assert verify_path_validity(grid, path, start, end), "Path should be valid"
        assert len(path) == 7, f"Expected path length 7, got {len(path)}"
        assert calculate_path_cost(grid, path) == 6, "Path cost should be 6"
    
    def test_path_around_obstacles(self):
        """Test pathfinding around obstacles."""
        grid = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]
        
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (4, 4)
        
        path = astar.find_path(start, end)
        
        assert path is not None, "Path should exist around obstacles"
        assert verify_path_validity(grid, path, start, end), "Path should be valid"
        # Minimum path must go around obstacles
        assert len(path) >= 9, "Path must go around obstacles"
    
    def test_weighted_grid_optimal_path(self):
        """Test that path prefers lower-cost cells on weighted grid."""
        grid = [
            [1, 10, 10, 1],
            [1, 10, 10, 1],
            [1, 1, 1, 1]
        ]
        
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 3)
        
        path = astar.find_path(start, end)
        
        assert path is not None, "Path should exist"
        assert verify_path_validity(grid, path, start, end), "Path should be valid"
        
        # Optimal path goes down then right (cost: 1+1+1+1+1 = 5)
        # Alternative through top-right would cost: 10+10+1+1+1 = 23
        path_cost = calculate_path_cost(grid, path)
        assert path_cost == 5, f"Expected optimal cost 5, got {path_cost}"
        
        # Verify the path doesn't go through expensive cells
        for pos in path:
            if pos != start:
                assert grid[pos[0]][pos[1]] == 1, "Path should avoid expensive cells"
    
    def test_no_path_exists(self):
        """Test when no path exists (fully blocked)."""
        grid = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ]
        
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 2)
        
        path = astar.find_path(start, end)
        
        assert path is None, "No path should exist when fully blocked"
    
    def test_start_equals_end(self):
        """Test when start position equals end position."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        
        astar = AStarGrid(grid)
        start = (1, 1)
        end = (1, 1)
        
        path = astar.find_path(start, end)
        
        assert path == [start], f"Expected [{start}], got {path}"
        assert len(path) == 1, "Path should contain only the start position"
    
    def test_invalid_coordinates(self):
        """Test that invalid coordinates raise ValueError."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        
        astar = AStarGrid(grid)
        
        # Test invalid start
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (2, 2))
        
        with pytest.raises(ValueError):
            astar.find_path((0, -1), (2, 2))
        
        with pytest.raises(ValueError):
            astar.find_path((3, 2), (2, 2))
        
        with pytest.raises(ValueError):
            astar.find_path((2, 3), (2, 2))
        
        # Test invalid end
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (-1, 0))
        
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (0, -1))
        
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (3, 2))
        
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (2, 3))
    
    def test_start_is_wall(self):
        """Test when start position is a wall."""
        grid = [
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        
        astar = AStarGrid(grid)
        
        path = astar.find_path((0, 0), (2, 2))
        assert path is None, "Should return None when start is a wall"
    
    def test_end_is_wall(self):
        """Test when end position is a wall."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 0]
        ]
        
        astar = AStarGrid(grid)
        
        path = astar.find_path((0, 0), (2, 2))
        assert path is None, "Should return None when end is a wall"
    
    def test_optimality_complex_weighted_grid(self):
        """Test optimality on a more complex weighted grid."""
        grid = [
            [1, 5, 1, 1],
            [1, 1, 1, 1],
            [1, 5, 5, 1]
        ]
        
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 3)
        
        path = astar.find_path(start, end)
        
        assert path is not None, "Path should exist"
        assert verify_path_validity(grid, path, start, end), "Path should be valid"
        
        path_cost = calculate_path_cost(grid, path)
        # Optimal path: (0,0)->(1,0)->(1,1)->(1,2)->(1,3)->(2,3) = 1+1+1+1+1 = 5
        assert path_cost == 5, f"Expected optimal cost 5, got {path_cost}"