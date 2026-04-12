"""
A* Pathfinding on a Weighted 2D Grid

This module implements the A* search algorithm for finding the shortest-cost
path between two points on a 2D grid where each cell has a movement cost.
"""

import heapq
from typing import List, Tuple, Optional


class AStarGrid:
    """
    A* pathfinding implementation for a weighted 2D grid.
    
    Uses Manhattan distance as the heuristic and supports 4-directional movement.
    """
    
    # Direction vectors: (row_delta, col_delta)
    DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the A* pathfinder with a 2D grid.
        
        Args:
            grid: 2D list where 0 = impassable wall, positive int = movement cost.
        
        Raises:
            ValueError: If grid is empty or rows have inconsistent lengths.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        
        self._rows = len(grid)
        self._cols = len(grid[0])
        
        # Validate grid consistency
        for row in grid:
            if len(row) != self._cols:
                raise ValueError("All rows must have the same length")
        
        self._grid = grid
    
    def _is_valid(self, row: int, col: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= row < self._rows and 0 <= col < self._cols
    
    def _is_wall(self, row: int, col: int) -> bool:
        """Check if a cell is impassable (cost of 0)."""
        return self._grid[row][col] == 0
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _calculate_path_cost(self, path: List[Tuple[int, int]]) -> int:
        """Calculate total cost of a given path."""
        return sum(self._grid[row][col] for row, col in path)
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest-cost path from start to end using A* algorithm.
        
        Args:
            start: Starting position as (row, col) tuple.
            end: Ending position as (row, col) tuple.
        
        Returns:
            List of (row, col) tuples forming the optimal path from start to end,
            or None if no path exists.
        
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate start coordinates
        if not self._is_valid(start[0], start[1]):
            raise ValueError(f"Start position {start} is out of bounds")
        if not self._is_valid(end[0], end[1]):
            raise ValueError(f"End position {end} is out of bounds")
        
        # Check if start or end is a wall
        if self._is_wall(start[0], start[1]):
            return None
        if self._is_wall(end[0], end[1]):
            return None
        
        # Handle start == end case
        if start == end:
            return [start]
        
        # Priority queue: (f_score, counter, row, col)
        # counter used to break ties consistently
        counter = 0
        open_set = [(self._manhattan_distance(start, end), counter, start[0], start[1])]
        
        # Track visited cells and their best g_scores
        came_from: dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        g_score: dict[Tuple[int, int], int] = {start: self._grid[start[0]][start[1]]}
        
        while open_set:
            counter += 1
            _, _, current_row, current_col = heapq.heappop(open_set)
            current = (current_row, current_col)
            
            # Found the destination
            if current == end:
                return self._reconstruct_path(came_from, current)
            
            # Explore neighbors
            for d_row, d_col in self.DIRECTIONS:
                neighbor_row, neighbor_col = current_row + d_row, current_col + d_col
                neighbor = (neighbor_row, neighbor_col)
                
                # Skip invalid or wall cells
                if not self._is_valid(neighbor_row, neighbor_col):
                    continue
                if self._is_wall(neighbor_row, neighbor_col):
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + self._grid[neighbor_row][neighbor_col]
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._manhattan_distance(neighbor, end)
                    heapq.heappush(open_set, (f_score, counter, neighbor_row, neighbor_col))
        
        # No path found
        return None
    
    def _reconstruct_path(
        self, 
        came_from: dict[Tuple[int, int], Optional[Tuple[int, int]]], 
        current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to end using the came_from dictionary."""
        path = []
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path


# =============================================================================
# PYTEST TESTS
# =============================================================================

import pytest


class TestAStarGrid:
    """Test suite for A* pathfinding implementation."""
    
    def test_simple_path_uniform_grid(self):
        """Test simple path on a uniform grid with no obstacles."""
        grid = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        
        path = astar.find_path((0, 0), (4, 4))
        
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)
        # Verify path is valid (adjacent cells)
        for i in range(len(path) - 1):
            row_diff = abs(path[i][0] - path[i+1][0])
            col_diff = abs(path[i][1] - path[i+1][1])
            assert (row_diff == 1 and col_diff == 0) or (row_diff == 0 and col_diff == 1)
        # Optimal path length for Manhattan distance 8 is 9 cells (8 moves)
        assert len(path) == 9
        # Cost should be 9 (9 cells * cost 1)
        assert sum(grid[r][c] for r, c in path) == 9
    
    def test_path_around_obstacles(self):
        """Test path finding with obstacles blocking direct routes."""
        grid = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        
        path = astar.find_path((0, 0), (4, 4))
        
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)
        # Verify no path cell is a wall
        for r, c in path:
            assert grid[r][c] != 0
        # Path should go around obstacles
        # Optimal: right, down, down, right, right, down, down
        assert len(path) == 8
    
    def test_weighted_grid_prefers_lower_cost(self):
        """Test that path prefers lower-cost cells when available."""
        grid = [
            [1, 1, 1, 1, 1],
            [1, 1, 5, 5, 1],
            [1, 1, 1, 1, 1],
            [1, 5, 5, 1, 1],
            [1, 1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        
        path = astar.find_path((0, 0), (4, 4))
        
        assert path is not None
        # Path should avoid high-cost cells (5s) and use the 1s
        # Optimal path: go through the middle corridor
        expected_cost = 9  # 9 cells at cost 1 each
        actual_cost = sum(grid[r][c] for r, c in path)
        assert actual_cost == expected_cost
        # Verify path doesn't use high-cost cells unnecessarily
        for r, c in path:
            if (r, c) in [(1, 2), (1, 3), (3, 1), (3, 2)]:
                pytest.fail(f"Path should not include high-cost cell ({r}, {c})")
    
    def test_no_path_exists(self):
        """Test when grid is completely blocked between start and end."""
        grid = [
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        
        path = astar.find_path((0, 0), (0, 4))
        
        assert path is None
    
    def test_start_equals_end(self):
        """Test when start and end positions are the same."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        
        path = astar.find_path((1, 1), (1, 1))
        
        assert path is not None
        assert path == [(1, 1)]
        assert len(path) == 1
    
    def test_invalid_coordinates(self):
        """Test handling of out-of-bounds coordinates."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Test start out of bounds
        with pytest.raises(ValueError):
            astar.find_path((5, 5), (1, 1))
        
        # Test end out of bounds
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (5, 5))
        
        # Test both out of bounds
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (5, 5))
    
    def test_wall_at_start_or_end(self):
        """Test when start or end position is a wall."""
        grid = [
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0]
        ]
        astar = AStarGrid(grid)
        
        # Start is a wall
        path = astar.find_path((0, 0), (2, 2))
        assert path is None
        
        # End is a wall
        path = astar.find_path((0, 1), (2, 2))
        assert path is None
    
    def test_optimality_on_weighted_grid(self):
        """Explicitly verify optimal path cost on weighted grid."""
        grid = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Test diagonal path (should still use Manhattan)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is not None
        # Manhattan distance is 4, so path has 5 cells
        assert len(path) == 5
        # Cost should be 5 (5 cells * cost 1)
        cost = sum(grid[r][c] for r, c in path)
        assert cost == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
