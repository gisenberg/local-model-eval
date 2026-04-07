from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.
    
    Grid values represent movement cost:
    - 0: impassable wall
    - positive int: cost to enter that cell
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D grid.
        
        Args:
            grid: 2D list where each value represents movement cost (0 = wall, positive = cost)
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
        
        Returns list of (neighbor_position, cost_to_enter_neighbor) tuples.
        """
        row, col = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        neighbors = []
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            new_pos = (new_row, new_col)
            
            if self._is_valid(new_pos):
                cost = self.grid[new_row][new_col]
                if cost > 0:  # Only consider passable cells
                    neighbors.append((new_pos, cost))
        
        return neighbors
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting position as (row, col) tuple
            end: Ending position as (row, col) tuple
            
        Returns:
            List of (row, col) coordinates representing the optimal path, or None if no path exists
            
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
        
        # A* algorithm
        # Priority queue: (f_score, g_score, position)
        open_set = [(0, 0, start)]
        
        # Track g_score (cost from start) for each position
        g_scores = {start: 0}
        
        # Track parent for path reconstruction
        came_from = {}
        
        while open_set:
            f_score, g_score, current = heapq.heappop(open_set)
            
            # Check if we reached the goal
            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]  # Reverse to get start to end
            
            # Explore neighbors
            for neighbor, move_cost in self._get_neighbors(current):
                tentative_g_score = g_score + move_cost
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self._heuristic(neighbor, end)
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))
        
        # No path found
        return None
    
    def _calculate_path_cost(self, path: List[Tuple[int, int]]) -> int:
        """Calculate total cost of a path."""
        if len(path) <= 1:
            return 0
        
        total_cost = 0
        for i in range(1, len(path)):
            row, col = path[i]
            total_cost += self.grid[row][col]
        
        return total_cost

import pytest
from typing import List, Tuple, Optional
import heapq

# Import the AStarGrid class
  # Replace with actual module name

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
    assert len(path) == 5  # Minimum path length
    
    # Verify path validity
    for i in range(len(path) - 1):
        curr = path[i]
        next_pos = path[i + 1]
        # Check adjacency
        assert abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1]) == 1
    
    # Verify optimality (cost should be 4 for 4 moves of cost 1 each)
    assert astar._calculate_path_cost(path) == 4

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
    
    # Verify path doesn't go through walls
    for pos in path:
        assert grid[pos[0]][pos[1]] != 0
    
    # Verify path validity
    for i in range(len(path) - 1):
        curr = path[i]
        next_pos = path[i + 1]
        assert abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1]) == 1

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
    
    # The optimal path should go through the middle row (cost 1 each)
    # rather than the top/bottom rows (cost 10 each)
    expected_cost = 1 + 1 + 1 + 1  # 4 moves of cost 1
    assert astar._calculate_path_cost(path) == expected_cost
    
    # Verify the path doesn't use high-cost cells
    for pos in path[1:]:  # Skip start position
        assert grid[pos[0]][pos[1]] != 10

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
    grid_with_wall = [
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar_wall = AStarGrid(grid_with_wall)
    path = astar_wall.find_path((0, 0), (2, 2))
    assert path is None
    
    # Test end on wall
    path = astar_wall.find_path((1, 1), (0, 0))
    assert path is None