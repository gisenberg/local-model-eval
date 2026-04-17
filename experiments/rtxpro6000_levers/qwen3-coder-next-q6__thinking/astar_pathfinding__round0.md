Here's a complete implementation of A* pathfinding on a weighted 2D grid with all the requested features and comprehensive pytest tests.

```python
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
                  0 indicates an impassable wall.
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
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols
    
    def _is_passable(self, pos: Tuple[int, int]) -> bool:
        """Check if cell at position is passable (not a wall)."""
        r, c = pos
        return self.grid[r][c] > 0
    
    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two points.
        
        Args:
            a: First point (row, col)
            b: Second point (row, col)
            
        Returns:
            Manhattan distance (|r1-r2| + |c1-c2|)
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
        r, c = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        neighbors = []
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self._in_bounds((nr, nc)) and self._is_passable((nr, nc)):
                neighbors.append((nr, nc))
        
        return neighbors
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: Starting position (row, col)
            end: Ending position (row, col)
            
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
            return [start]
        
        # Check if start or end is a wall
        if not self._is_passable(start) or not self._is_passable(end):
            return None
        
        # A* algorithm
        # Priority queue: (f_score, g_score, position)
        open_set = [(self._manhattan_distance(start, end), 0, start)]
        
        # Track best g_score to each node
        g_score = {start: 0}
        
        # Track path for reconstruction
        came_from = {}
        
        # Set for O(1) lookup of visited nodes
        closed_set = set()
        
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            
            # Skip if we've already found a better path to this node
            if current in closed_set:
                continue
            
            # Found the goal
            if current == end:
                # Reconstruct path
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                path.reverse()
                return path
            
            closed_set.add(current)
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Cost to move to neighbor is the neighbor's grid value
                tentative_g = current_g + self.grid[neighbor[0]][neighbor[1]]
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._manhattan_distance(neighbor, end)
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        # No path found
        return None


# Helper function to calculate path cost
def calculate_path_cost(path: List[Tuple[int, int]], grid: List[List[int]]) -> int:
    """Calculate total cost of a path (excluding start cell cost)."""
    if len(path) <= 1:
        return 0
    total = 0
    for pos in path[1:]:  # Skip start cell
        r, c = pos
        total += grid[r][c]
    return total


# Helper function to validate path
def is_valid_path(path: List[Tuple[int, int]], grid: List[List[int]], 
                  start: Tuple[int, int], end: Tuple[int, int]) -> bool:
    """Validate that a path is valid (adjacent cells, within bounds, not walls)."""
    if not path:
        return False
    
    if path[0] != start or path[-1] != end:
        return False
    
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        
        # Check adjacency (Manhattan distance should be 1)
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return False
        
        # Check bounds and passability
        if not (0 <= r2 < len(grid) and 0 <= c2 < len(grid[0])):
            return False
        if grid[r2][c2] == 0:
            return False
    
    return True


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
    assert is_valid_path(path, grid, (0, 0), (2, 2))
    
    # On uniform grid, Manhattan path length should be 4 steps (5 cells)
    assert len(path) == 5
    
    # Cost should be 4 (4 cells with cost 1 each)
    assert calculate_path_cost(path, grid) == 4


def test_path_around_obstacles():
    """Test pathfinding around obstacles."""
    grid = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    
    path = astar.find_path((0, 0), (4, 4))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
    assert is_valid_path(path, grid, (0, 0), (4, 4))
    
    # Cost should be 8 (8 cells with cost 1 each)
    assert calculate_path_cost(path, grid) == 8


def test_weighted_grid():
    """Test that path prefers lower-cost cells."""
    grid = [
        [1, 1, 1, 1, 1],
        [1, 10, 10, 10, 1],
        [1, 10, 10, 10, 1],
        [1, 10, 10, 10, 1],
        [1, 1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    
    path = astar.find_path((0, 0), (4, 4))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
    assert is_valid_path(path, grid, (0, 0), (4, 4))
    
    # Path should go around the high-cost area (cost 8)
    # vs going through the high-cost area (cost 40+)
    cost = calculate_path_cost(path, grid)
    assert cost == 8  # Should take the cheaper outer path


def test_no_path_exists():
    """Test when no path exists due to obstacles."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    # Center is blocked, but corners are connected via edges
    # Let's make a truly blocked case
    grid_blocked = [
        [1, 1, 1],
        [1, 0, 0],  # Right side blocked
        [1, 0, 1]
    ]
    astar = AStarGrid(grid_blocked)
    
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
    assert calculate_path_cost(path, grid) == 0


def test_invalid_coordinates():
    """Test handling of invalid coordinates."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Test out-of-bounds start
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    
    # Test out-of-bounds end
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (3, 3))
    
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Key Features of the Implementation:

1. **Robust A* Algorithm**: Uses a min-heap priority queue with proper handling of duplicate nodes via `g_score` tracking.

2. **Manhattan Heuristic**: Uses `|r1-r2| + |c1-c2|` as the heuristic, which is admissible for 4-directional movement.

3. **Edge Case Handling**:
   - Start == end returns `[start]`
   - Start/end on walls returns `None`
   - Out-of-bounds coordinates raise `ValueError`
   - Empty grid validation

4. **Optimality**: The algorithm guarantees the minimum-cost path by using the correct A* formulation with `f = g + h`.

5. **Type Hints**: Comprehensive type annotations throughout.

6. **Comprehensive Tests**: 6 test cases covering all requirements:
   - Simple uniform grid pathfinding
   - Path around obstacles
   - Weighted grid preference for lower-cost paths
   - No-path scenarios
   - Start equals end
   - Invalid coordinates and wall handling

The implementation correctly handles weighted grids where the cost to enter a cell is stored in the grid, and the path cost is calculated as the sum of all cell costs *except* the starting cell (since you don't pay to enter your starting position).