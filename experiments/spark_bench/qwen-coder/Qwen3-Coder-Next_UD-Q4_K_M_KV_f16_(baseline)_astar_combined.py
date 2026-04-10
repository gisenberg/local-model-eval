import heapq
from typing import List, Tuple, Optional
import math

class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.
    
    Grid values represent movement cost:
    - 0 = impassable wall
    - Positive integer = cost to enter that cell
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid with movement costs.
        
        Args:
            grid: 2D list where each cell contains the cost to enter that cell.
                  0 represents an impassable wall.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # Validate grid dimensions are consistent
        for row in grid:
            if len(row) != self.cols:
                raise ValueError("All rows must have the same length")
    
    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _is_passable(self, pos: Tuple[int, int]) -> bool:
        """Check if cell at position is passable (not a wall)."""
        row, col = pos
        return self.grid[row][col] > 0
    
    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two points.
        
        Args:
            a: First point (row, col)
            b: Second point (row, col)
            
        Returns:
            Manhattan distance between a and b
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighboring cells (4-directional movement).
        
        Args:
            pos: Current position (row, col)
            
        Returns:
            List of valid neighbor positions
        """
        row, col = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        neighbors = []
        
        for dr, dc in directions:
            new_pos = (row + dr, col + dc)
            if self._in_bounds(new_pos) and self._is_passable(new_pos):
                neighbors.append(new_pos)
        
        return neighbors
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: Starting position (row, col)
            end: Target position (row, col)
            
        Returns:
            List of positions from start to end inclusive, or None if no path exists.
        """
        # Validate coordinates are in bounds
        if not self._in_bounds(start):
            raise ValueError(f"Start position {start} is out of bounds")
        if not self._in_bounds(end):
            raise ValueError(f"End position {end} is out of bounds")
        
        # Handle start == end case
        if start == end:
            if self._is_passable(start):
                return [start]
            else:
                return None
        
        # Check if start or end is a wall
        if not self._is_passable(start) or not self._is_passable(end):
            return None
        
        # A* algorithm
        # Priority queue: (f_score, counter, position)
        # counter is used to break ties (FIFO for same f_score)
        counter = 0
        open_set = [(self._manhattan_distance(start, end), counter, start)]
        
        # Track best g_score to each position
        g_score = {start: 0}
        
        # For reconstructing path: came_from[pos] = previous position
        came_from = {}
        
        # Set for O(1) lookup of open set positions
        open_set_hash = {start}
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)
            
            # If we reached the end, reconstruct path
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                # Cost to move from current to neighbor
                cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g_score = g_score[current] + cost
                
                # If this path to neighbor is better, update
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self._manhattan_distance(neighbor, end)
                    
                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f_score, counter, neighbor))
                        open_set_hash.add(neighbor)
        
        # No path found
        return None


# Helper function to calculate path cost
def calculate_path_cost(path: List[Tuple[int, int]], grid: List[List[int]]) -> int:
    """Calculate total cost of a path (excluding start cell cost)."""
    if len(path) <= 1:
        return 0
    total = 0
    for pos in path[1:]:  # Skip start cell
        row, col = pos
        total += grid[row][col]
    return total


# Pytest tests
import pytest

def test_simple_path_uniform_grid():
    """Test basic pathfinding on a uniform grid."""
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
    
    # Manhattan distance is 4, and each step costs 1, so optimal cost is 4
    assert calculate_path_cost(path, grid) == 4
    
    # Path length should be 5 (Manhattan distance + 1)
    assert len(path) == 5


def test_path_around_obstacles():
    """Test pathfinding when obstacles require going around."""
    grid = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Path from top-left to bottom-right, avoiding the wall in the middle
    path = astar.find_path((0, 0), (4, 4))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
    
    # Should go around the wall - Manhattan distance is 8, cost should be 8
    assert calculate_path_cost(path, grid) == 8


def test_weighted_grid():
    """Test that path prefers lower-cost cells."""
    # Create a grid where going through high-cost cells is suboptimal
    grid = [
        [1, 1, 1, 1, 1],
        [1, 10, 10, 10, 1],
        [1, 1, 1, 1, 1],
        [1, 10, 10, 10, 1],
        [1, 1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    
    path = astar.find_path((0, 0), (4, 4))
    assert path is not None
    
    # Should avoid the high-cost column (cost 10) and go through the low-cost path
    # Optimal path: go down first, then right, cost = 4 (down) + 4 (right) = 8
    cost = calculate_path_cost(path, grid)
    assert cost == 8
    
    # Verify path doesn't go through high-cost cells
    for pos in path:
        row, col = pos
        assert grid[row][col] != 10


def test_no_path_exists():
    """Test when start and end are separated by walls."""
    grid = [
        [1, 1, 1],
        [0, 0, 0],  # Wall blocking the way
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    
    path = astar.find_path((0, 0), (2, 2))
    assert path is None


def test_start_equals_end():
    """Test when start and end positions are the same."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Valid start/end
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]
    
    # Start/end on a wall
    grid_with_wall = [
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar_wall = AStarGrid(grid_with_wall)
    path = astar_wall.find_path((0, 1), (0, 1))
    assert path is None


def test_invalid_coordinates():
    """Test handling of invalid coordinates."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Start out of bounds
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    
    # End out of bounds
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (3, 3))
    
    # Both out of bounds
    with pytest.raises(ValueError):
        astar.find_path((10, 10), (20, 20))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
