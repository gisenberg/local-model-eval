from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.
    
    Grid values represent movement cost:
    - 0: impassable wall
    - positive int: cost to enter that cell
    
    Supports 4-directional movement with Manhattan distance heuristic.
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid with the given 2D cost matrix.
        
        Args:
            grid: 2D list where each value represents the cost to enter that cell
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
        Get valid neighboring cells and their movement costs.
        
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
                if cost > 0:  # Only include passable cells
                    neighbors.append((new_pos, cost))
        
        return neighbors
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
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
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end coordinates are out of bounds")
        
        # Handle edge cases
        if start == end:
            return [start]
        
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        
        # A* algorithm
        # Priority queue: (f_score, g_score, position, path)
        open_set = [(self._heuristic(start, end), 0, start, [start])]
        visited = set()
        
        while open_set:
            f_score, g_score, current, path = heapq.heappop(open_set)
            
            if current in visited:
                continue
            
            visited.add(current)
            
            if current == end:
                return path
            
            for neighbor, cost in self._get_neighbors(current):
                if neighbor in visited:
                    continue
                
                new_g_score = g_score + cost
                new_f_score = new_g_score + self._heuristic(neighbor, end)
                
                # Only add if we haven't visited this neighbor or found a better path
                if neighbor not in visited:
                    heapq.heappush(open_set, (new_f_score, new_g_score, neighbor, path + [neighbor]))
        
        return None  # No path found


# Test suite
def test_simple_path_uniform_grid():
    """Test simple path on uniform grid."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
    assert path[0] == (0, 0) and path[-1] == (2, 2)
    
    # Verify path validity
    for i in range(len(path) - 1):
        curr = path[i]
        next_pos = path[i + 1]
        assert abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1]) == 1

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
    assert path[0] == (0, 0) and path[-1] == (3, 3)
    
    # Verify no obstacles in path
    for pos in path:
        assert grid[pos[0]][pos[1]] != 0

def test_weighted_grid_optimal_path():
    """Test that algorithm prefers lower-cost cells."""
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    # Should prefer going around the expensive middle column
    expected_cost = 1 + 1 + 1 + 1 + 1  # 5 cells with cost 1 each
    actual_cost = sum(grid[pos[0]][pos[1]] for pos in path[1:])  # exclude start
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
    """Test handling of invalid coordinates."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Test out of bounds start
    try:
        astar.find_path((-1, 0), (2, 2))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test out of bounds end
    try:
        astar.find_path((0, 0), (5, 5))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test start on wall
    grid_with_wall = [
        [0, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar_wall = AStarGrid(grid_with_wall)
    path = astar_wall.find_path((0, 0), (2, 2))
    assert path is None