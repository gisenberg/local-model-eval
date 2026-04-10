import heapq
from typing import List, Tuple, Optional

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
        Initialize the AStarGrid with a 2D grid.
        
        Args:
            grid: 2D list where each value represents movement cost
                  (0 = wall, positive int = cost to enter cell)
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def _is_valid(self, row: int, col: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _get_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get valid 4-directional neighbors (up, down, left, right)."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = []
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if self._is_valid(new_row, new_col) and self.grid[new_row][new_col] > 0:
                neighbors.append((new_row, new_col))
        
        return neighbors
    
    def _heuristic(self, row: int, col: int, end_row: int, end_col: int) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(row - end_row) + abs(col - end_col)
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting coordinates (row, col)
            end: Ending coordinates (row, col)
            
        Returns:
            List of coordinates representing the optimal path, or None if no path exists
            
        Raises:
            ValueError: If start or end coordinates are out of bounds
        """
        # Validate coordinates
        if not self._is_valid(start[0], start[1]) or not self._is_valid(end[0], end[1]):
            raise ValueError("Start or end coordinates are out of bounds")
        
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        
        # Handle start == end case
        if start == end:
            return [start]
        
        # A* algorithm
        # Priority queue: (f_score, g_score, current_node, path)
        open_set = [(0, 0, start, [start])]
        visited = set()
        
        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)
            
            # If we reached the end
            if current == end:
                return path
            
            # Skip if already visited
            if current in visited:
                continue
            
            visited.add(current)
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current[0], current[1]):
                if neighbor in visited:
                    continue
                
                # Calculate cost to move to neighbor
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                new_g_score = g_score + move_cost
                new_f_score = new_g_score + self._heuristic(neighbor[0], neighbor[1], end[0], end[1])
                
                # Add to open set
                new_path = path + [neighbor]
                heapq.heappush(open_set, (new_f_score, new_g_score, neighbor, new_path))
        
        # No path found
        return None

import pytest
from typing import List, Tuple, Optional

# Import the AStarGrid class
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
        curr = path[i]
        next_pos = path[i + 1]
        # Check adjacency
        assert abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1]) == 1
        # Check cost
        assert grid[curr[0]][curr[1]] > 0
        assert grid[next_pos[0]][next_pos[1]] > 0

def test_path_around_obstacles():
    """Test path finding around obstacles."""
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (3, 3))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (3, 3)
    
    # Verify path doesn't go through walls
    for pos in path:
        assert grid[pos[0]][pos[1]] > 0

def test_weighted_grid_optimality():
    """Test that algorithm prefers lower-cost paths."""
    grid = [
        [1, 10, 1, 1],
        [1, 10, 1, 1],
        [1, 10, 1, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (3, 3))
    
    assert path is not None
    
    # Calculate total cost
    total_cost = sum(grid[pos[0]][pos[1]] for pos in path[1:])  # Don't count start
    
    # The optimal path should avoid the high-cost column
    # Expected path: (0,0) -> (1,0) -> (2,0) -> (3,0) -> (3,1) -> (3,2) -> (3,3)
    # Cost: 1+1+1+1+1+1 = 6
    assert total_cost == 6

def test_no_path_exists():
    """Test when no path exists due to complete blockage."""
    grid = [
        [1, 1, 1],
        [0, 0, 0],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is None

def test_start_equals_end():
    """Test when start and end are the same position."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    
    assert path == [(1, 1)]

def test_invalid_coordinates():
    """Test invalid coordinate handling."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Test out of bounds start
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    
    # Test out of bounds end
    with pytest.raises(ValueError):
        astar.find_path((1, 1), (5, 5))
    
    # Test start on wall
    grid[1][1] = 0
    path = astar.find_path((1, 1), (2, 2))
    assert path is None
    
    # Test end on wall
    grid[2][2] = 0
    path = astar.find_path((0, 0), (2, 2))
    assert path is None