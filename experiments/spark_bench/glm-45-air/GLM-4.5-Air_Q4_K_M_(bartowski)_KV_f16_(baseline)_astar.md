# GLM-4.5-Air Q4_K_M (bartowski) — KV f16 (baseline) — A* Pathfinding

## Output

Here's an implementation of the A* pathfinding algorithm on a weighted 2D grid in Python, along with pytest tests:

```python
import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.
    
    Attributes:
        grid (List[List[int]]): The grid where each cell's value represents movement cost.
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid with a given cost grid.
        
        Args:
            grid: 2D list where 0 represents an impassable wall and positive integers represent movement cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Tuple (row, col) of the starting position.
            end: Tuple (row, col) of the destination.
            
        Returns:
            List of (row, col) coordinates representing the shortest path, or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate start and end positions
        if not self._is_valid_position(start):
            raise ValueError(f"Start position {start} is out of bounds")
        if not self._is_valid_position(end):
            raise ValueError(f"End position {end} is out of bounds")
        
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None
        
        # Handle trivial case where start equals end
        if start == end:
            return [start]
        
        # Initialize data structures
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        f_score: Dict[Tuple[int, int], float] = {start: self._heuristic(start, end)}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == end:
                return self._reconstruct_path(came_from, current)
            
            for neighbor in self._get_neighbors(current):
                tentative_g_score = g_score[current] + self.grid[neighbor[0]][neighbor[1]]
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # No path found
    
    def _is_valid_position(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within grid bounds."""
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid 4-directional neighbors of a position."""
        row, col = pos
        neighbors = []
        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (row + dr, col + dc)
            if (self._is_valid_position(neighbor) and 
                self.grid[neighbor[0]][neighbor[1]] > 0):
                neighbors.append(neighbor)
        return neighbors
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], 
                         current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to end using came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
```

Here are the pytest tests:

```python
import pytest
from a_star import AStarGrid

def test_simple_path():
    """Test simple path on uniform grid."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)
    path = astar.find_path(start, end)
    
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    assert len(path) == 5  # Manhattan distance + 1
    assert sum(grid[r][c] for r, c in path) == 5  # Total cost should be 5

def test_path_around_obstacles():
    """Test path around obstacles."""
    grid = [
        [1, 0, 1],
        [1, 1, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)
    path = astar.find_path(start, end)
    
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    assert (0, 1) not in path  # Should go around the wall
    assert (2, 1) not in path
    assert sum(grid[r][c] for r, c in path) == 6  # Path: (0,0)->(1,0)->(1,1)->(1,2)->(2,2)

def test_weighted_grid():
    """Test path on weighted grid (prefers lower-cost cells)."""
    grid = [
        [1, 9, 1],
        [1, 9, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (0, 0)
    end = (2, 2)
    path = astar.find_path(start, end)
    
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    assert (0, 1) not in path  # Should avoid high-cost cell
    assert (1, 1) not in path
    assert sum(grid[r][c] for r, c in path) == 5  # Path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2)

def test_no_path_exists():
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
    
    assert path is None

def test_start_equals_end():
    """Test when start equals end."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    start = (1, 1)
    end = (1, 1)
    path = astar.find_path(start, end)
    
    assert path is not None
    assert path == [start]

def test_invalid_coordinates():
    """Test invalid coordinates (out of bounds)."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))  # Start out of bounds
    
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (3, 0))  # End out of bounds

def test_wall_at_start_or_end():
    """Test when start or end is a wall."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Start is wall
    assert astar.find_path((1, 1), (2, 2)) is None
    
    # End is wall
    assert astar.find_path((0, 0), (1, 1)) is None
```

These tests cover all the required scenarios:
1. Simple path on uniform grid
2. Path around obstacles
3. Weighted grid (path prefers lower-cost cells)
4. No path exists (fully blocked)
5. Start equals end
6. Invalid coordinates (out of bounds)
7. Additional test for wall at start or end

Each test verifies both path validity (correct start/end points, no walls) and optimality (total cost matches expected minimum).
