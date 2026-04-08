from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    """A* pathfinding on a weighted 2D grid."""
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a weighted 2D grid.
        
        Args:
            grid: 2D list where grid[r][c] represents movement cost
                  (0 = impassable wall, positive int = cost to enter cell)
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting coordinates (row, col)
            end: Ending coordinates (row, col)
            
        Returns:
            List of (row, col) coordinates representing the path, or None if no path exists
            
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
        # Priority queue: (f_score, g_score, current_node)
        open_set = [(0, 0, start)]
        g_score = {start: 0}  # Actual cost from start
        parent = {}  # For path reconstruction
        closed_set = set()  # Processed nodes
        
        while open_set:
            f_score, g_score_current, current = heapq.heappop(open_set)
            
            # Skip if we've already processed this node with a better cost
            if current in closed_set:
                continue
                
            closed_set.add(current)
            
            # Found the goal
            if current == end:
                return self._reconstruct_path(parent, current)
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                    
                neighbor_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g_score = g_score[current] + neighbor_cost
                
                # If this path to neighbor is better, update it
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    g_score[neighbor] = tentative_g_score
                    h_score = self._manhattan_distance(neighbor, end)
                    f_score_new = tentative_g_score + h_score
                    parent[neighbor] = current
                    heapq.heappush(open_set, (f_score_new, tentative_g_score, neighbor))
        
        # No path found
        return None
    
    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid 4-directional neighbors of a position."""
        row, col = pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # up, down, left, right
            new_row, new_col = row + dr, col + dc
            if self._is_valid((new_row, new_col)) and self.grid[new_row][new_col] > 0:
                neighbors.append((new_row, new_col))
        return neighbors
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _reconstruct_path(self, parent: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from parent pointers."""
        path = [current]
        while current in parent:
            current = parent[current]
            path.append(current)
        return path[::-1]  # Reverse to get path from start to end


# Test suite
import pytest

def test_simple_path_uniform_grid():
    """Test simple path on uniform grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Path should be optimal - minimum 4 steps in any direction
    assert len(path) == 5

def test_path_around_obstacles():
    """Test path finding around obstacles."""
    grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Should avoid the wall at (1, 1)
    assert (1, 1) not in path

def test_weighted_grid_optimality():
    """Test that algorithm prefers lower-cost cells."""
    grid = [[1, 10, 1], [1, 1, 1], [1, 10, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    # Should prefer the path through the middle row (cost 1 each) over top/bottom (cost 10)
    total_cost = sum(grid[r][c] for r, c in path[1:])  # Exclude start cell
    assert total_cost == 4  # 4 cells with cost 1 each

def test_no_path_exists():
    """Test when no path exists due to complete blockage."""
    grid = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
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
    
    # Out of bounds start
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    
    # Out of bounds end
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))