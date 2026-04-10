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
        Initialize the AStarGrid with a 2D grid of movement costs.
        
        Args:
            grid: 2D list where each value represents the cost to enter that cell
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not a wall."""
        row, col = pos
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        return self.grid[row][col] > 0
    
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
            List of (row, col) coordinates representing the optimal path,
            or None if no path exists
            
        Raises:
            ValueError: If start or end coordinates are out of bounds
        """
        # Validate coordinates
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start position {start} is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End position {end} is out of bounds")
        
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        
        # Handle start == end case
        if start == end:
            return [start]
        
        # A* algorithm
        # Priority queue: (f_score, g_score, position)
        open_set = [(self._heuristic(start, end), 0, start)]
        
        # Track best g_score for each position
        g_scores = {start: 0}
        
        # Track parent positions for path reconstruction
        parents = {}
        
        while open_set:
            f_score, g_score, current = heapq.heappop(open_set)
            
            # Found the goal
            if current == end:
                # Reconstruct path
                path = []
                while current in parents:
                    path.append(current)
                    current = parents[current]
                path.append(start)
                return path[::-1]
            
            # Skip if we've found a better path to this position
            if g_score > g_scores.get(current, float('inf')):
                continue
            
            # Explore neighbors
            for neighbor, move_cost in self._get_neighbors(current):
                tentative_g = g_score + move_cost
                
                if tentative_g < g_scores.get(neighbor, float('inf')):
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    parents[neighbor] = current
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        # No path found
        return None

import pytest
from typing import List, Tuple, Optional
import heapq

# Import the AStarGrid class from the previous code
# (In a real scenario, this would be imported from the module)

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
        curr, next_pos = path[i], path[i + 1]
        assert abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1]) == 1
    
    # Verify optimality (total cost should be 4)
    total_cost = sum(grid[r][c] for r, c in path[1:])  # Don't count start
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
    
    # Verify path doesn't go through walls
    for r, c in path:
        assert grid[r][c] != 0
    
    # Verify path validity
    for i in range(len(path) - 1):
        curr, next_pos = path[i], path[i + 1]
        assert abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1]) == 1

def test_weighted_grid_optimal_path():
    """Test that algorithm prefers lower-cost cells in weighted grid."""
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
    
    # The optimal path should go through the middle (cost 1 cells)
    # rather than the expensive side paths (cost 10 cells)
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4  # Should use the cheaper middle path
    
    # Verify path doesn't use expensive cells unnecessarily
    expensive_cells_used = sum(1 for r, c in path if grid[r][c] == 10)
    assert expensive_cells_used == 0

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
    """Test when start and end positions are the same."""
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
        astar.find_path((-1, 0), (1, 1))
    
    # Test out of bounds end
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))
    
    # Test start on wall
    grid_with_wall = [
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar_wall = AStarGrid(grid_with_wall)
    path = astar_wall.find_path((0, 0), (2, 2))
    assert path is None