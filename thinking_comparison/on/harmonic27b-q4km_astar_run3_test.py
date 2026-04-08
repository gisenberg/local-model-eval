from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize A* pathfinding on a weighted 2D grid.
        
        Args:
            grid: 2D list where grid[i][j] represents movement cost (0 = wall, positive = cost)
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def _heuristic(self, current: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(current[0] - goal[0]) + abs(current[1] - goal[1])
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid 4-directional neighbors."""
        row, col = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        neighbors = []
        
        for dr, dc in directions:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                if self.grid[new_row][new_col] > 0:  # Not a wall
                    neighbors.append((new_row, new_col))
        
        return neighbors
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting position (row, col)
            end: Ending position (row, col)
            
        Returns:
            List of coordinates from start to end inclusive, or None if no path exists
            
        Raises:
            ValueError: If start or end coordinates are out of bounds
        """
        # Validate coordinates
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start position {start} is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End position {end} is out of bounds")
        
        # Handle edge cases
        if start == end:
            return [start]
        
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        
        # A* algorithm
        # Priority queue: (f_score, g_score, position)
        open_set = [(self._heuristic(start, end), 0, start)]
        
        # Track g_scores (actual cost from start)
        g_scores = {start: 0}
        
        # Track came_from for path reconstruction
        came_from = {}
        
        while open_set:
            # Get node with lowest f_score
            f_score, g_score, current = heapq.heappop(open_set)
            
            # Found goal
            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]  # Reverse to get start to end
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                # Cost to move to neighbor
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g_score = g_score + move_cost
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_scores or tentative_g_score < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self._heuristic(neighbor, end)
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))
        
        # No path found
        return None


# Test cases
import pytest

def test_simple_path_uniform_grid():
    """Test simple path on uniform grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Minimum path length
    
    # Verify optimality: total cost should be 4
    total_cost = sum(grid[r][c] for r, c in path[1:])  # Exclude start, include end
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
        assert grid[r][c] > 0

def test_weighted_grid():
    """Test that algorithm prefers lower-cost cells."""
    grid = [
        [1, 10, 1],
        [1, 1, 1],
        [1, 10, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    
    # The optimal path should avoid the 10-cost cells
    # Path should be: (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2)
    # Total cost: 1 + 1 + 1 + 1 = 4
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4

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
    """Test when start equals end."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    
    assert path == [(0, 0)]

def test_invalid_coordinates():
    """Test invalid coordinate handling."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    # Test out of bounds start
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    
    # Test out of bounds end
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (3, 3))