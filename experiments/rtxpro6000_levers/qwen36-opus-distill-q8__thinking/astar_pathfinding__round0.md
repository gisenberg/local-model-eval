```python
"""
A* Pathfinding implementation on a weighted 2D grid.

This module provides an efficient pathfinding algorithm that finds the
shortest path (minimum total cost) between two points on a grid where
each cell has an associated movement cost.
"""

import heapq
from typing import List, Optional, Tuple


class AStarGrid:
    """A* pathfinding on a weighted 2D grid.
    
    Attributes:
        grid: 2D list where each value represents the cost to enter that cell.
              0 indicates an impassable wall.
        rows: Number of rows in the grid.
        cols: Number of columns in the grid.
    """
    
    # Direction vectors for 4-directional movement (up, down, left, right)
    DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def __init__(self, grid: List[List[int]]) -> None:
        """Initialize the A* grid with movement costs.
        
        Args:
            grid: 2D list of integers where 0 = wall, positive int = cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def _manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions.
        
        Args:
            pos1: First position as (row, col).
            pos2: Second position as (row, col).
            
        Returns:
            Manhattan distance between the two positions.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _is_valid_position(self, row: int, col: int) -> bool:
        """Check if a position is within grid bounds.
        
        Args:
            row: Row index to check.
            col: Column index to check.
            
        Returns:
            True if position is within bounds, False otherwise.
        """
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _is_passable(self, row: int, col: int) -> bool:
        """Check if a cell is passable (not a wall).
        
        Args:
            row: Row index of the cell.
            col: Column index of the cell.
            
        Returns:
            True if cell is passable (cost > 0), False if wall.
        """
        return self.grid[row][col] > 0
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """Find the shortest path from start to end using A* algorithm.
        
        Uses A* search with Manhattan distance heuristic to find the
        minimum-cost path through the grid.
        
        Args:
            start: Starting position as (row, col).
            end: Ending position as (row, col).
            
        Returns:
            List of (row, col) coordinates from start to end inclusive,
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate coordinates are within bounds
        if not self._is_valid_position(start[0], start[1]):
            raise ValueError(f"Start position {start} is out of bounds")
        if not self._is_valid_position(end[0], end[1]):
            raise ValueError(f"End position {end} is out of bounds")
        
        # Handle start == end case
        if start == end:
            return [start]
        
        # Check if start or end is a wall
        if not self._is_passable(start[0], start[1]) or not self._is_passable(end[0], end[1]):
            return None
        
        # A* algorithm state
        # g_score: actual cost from start to each node
        g_score: dict[Tuple[int, int], int] = {start: 0}
        # f_score: estimated total cost (g + h)
        f_score: dict[Tuple[int, int], int] = {start: self._manhattan_distance(start, end)}
        # parent: track path reconstruction
        parent: dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        # Open set: priority queue of (f_score, position)
        open_set: List[Tuple[int, Tuple[int, int]]] = [(f_score[start], start)]
        # Closed set: positions already evaluated
        closed_set: set[Tuple[int, int]] = set()
        
        while open_set:
            # Get position with lowest f_score
            _, current = heapq.heappop(open_set)
            
            # Skip if already processed
            if current in closed_set:
                continue
            
            # Check if we reached the goal
            if current == end:
                # Reconstruct path
                path = []
                node = current
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path
            
            # Mark current as processed
            closed_set.add(current)
            
            # Explore neighbors
            for dr, dc in self.DIRECTIONS:
                neighbor = (current[0] + dr, current[1] + dc)
                
                # Skip if out of bounds or not passable
                if not self._is_valid_position(neighbor[0], neighbor[1]):
                    continue
                if not self._is_passable(neighbor[0], neighbor[1]):
                    continue
                
                # Skip if already processed
                if neighbor in closed_set:
                    continue
                
                # Calculate new g_score (cost to reach neighbor through current)
                new_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]
                
                # Update if this path is better
                if neighbor not in g_score or new_g < g_score[neighbor]:
                    g_score[neighbor] = new_g
                    f = new_g + self._manhattan_distance(neighbor, end)
                    f_score[neighbor] = f
                    parent[neighbor] = current
                    heapq.heappush(open_set, (f, neighbor))
        
        # No path found
        return None
```

```python
"""
Pytest tests for AStarGrid pathfinding implementation.

Tests cover: simple paths, obstacle navigation, weighted costs,
blocked paths, start==end case, and invalid coordinates.
"""

import pytest
from astar_grid import AStarGrid


# Test fixtures
@pytest.fixture
def uniform_grid():
    """Create a uniform grid with cost 1 for all cells."""
    return AStarGrid([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]
    ])


@pytest.fixture
def obstacle_grid():
    """Create a grid with obstacles forming a wall."""
    return AStarGrid([
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ])


@pytest.fixture
def weighted_grid():
    """Create a grid with varying movement costs."""
    return AStarGrid([
        [1, 1, 1, 1, 1],
        [1, 5, 5, 5, 1],
        [1, 5, 1, 5, 1],
        [1, 5, 5, 5, 1],
        [1, 1, 1, 1, 1]
    ])


@pytest.fixture
def blocked_grid():
    """Create a grid where start and end are completely separated."""
    return AStarGrid([
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1],
        [0, 0, 0, 0, 0],
        [1, 1, 0, 0, 1],
        [1, 1, 0, 0, 1]
    ])


def calculate_path_cost(grid: AStarGrid, path: List[Tuple[int, int]]) -> int:
    """Calculate the total cost of a path."""
    return sum(grid.grid[r][c] for r, c in path)


def verify_path_validity(grid: AStarGrid, path: List[Tuple[int, int]]) -> bool:
    """Verify that a path is valid (all cells passable and connected)."""
    if not path:
        return False
    
    # Check all cells are passable
    for r, c in path:
        if not grid._is_passable(r, c):
            return False
    
    # Check all cells are connected (adjacent)
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return False
    
    return True


# Test 1: Simple path on uniform grid
def test_simple_path_uniform_grid(uniform_grid):
    """Test finding a simple path on a uniform cost grid."""
    start = (0, 0)
    end = (4, 4)
    
    path = uniform_grid.find_path(start, end)
    
    assert path is not None, "Path should exist on uniform grid"
    assert path[0] == start, "Path should start at start position"
    assert path[-1] == end, "Path should end at end position"
    assert verify_path_validity(uniform_grid, path), "Path should be valid"
    # On uniform grid, optimal path length is Manhattan distance + 1 (inclusive)
    expected_length = abs(start[0] - end[0]) + abs(start[1] - end[1]) + 1
    assert len(path) == expected_length, f"Path length should be {expected_length}"
    # Total cost equals number of cells (each costs 1)
    assert calculate_path_cost(uniform_grid, path) == expected_length


# Test 2: Path around obstacles
def test_path_around_obstacles(obstacle_grid):
    """Test navigating around obstacles in the grid."""
    start = (0, 0)
    end = (4, 4)
    
    path = obstacle_grid.find_path(start, end)
    
    assert path is not None, "Path should exist around obstacles"
    assert path[0] == start, "Path should start at start position"
    assert path[-1] == end, "Path should end at end position"
    assert verify_path_validity(obstacle_grid, path), "Path should be valid"
    
    # Verify path doesn't go through any obstacles
    for r, c in path:
        assert obstacle_grid.grid[r][c] > 0, f"Path should not go through obstacle at ({r}, {c})"


# Test 3: Weighted grid prefers lower-cost cells
def test_weighted_grid_prefers_low_cost(weighted_grid):
    """Test that A* prefers paths through lower-cost cells."""
    start = (0, 0)
    end = (4, 4)
    
    path = weighted_grid.find_path(start, end)
    
    assert path is not None, "Path should exist on weighted grid"
    assert path[0] == start, "Path should start at start position"
    assert path[-1] == end, "Path should end at end position"
    assert verify_path_validity(weighted_grid, path), "Path should be valid"
    
    # The optimal path should avoid the high-cost center (cost 5)
    # Path going around the perimeter should be cheaper
    path_cost = calculate_path_cost(weighted_grid, path)
    
    # Calculate cost of going through center (would be much higher)
    center_path_cost = 1 + 5 + 5 + 5 + 1 + 5 + 5 + 5 + 1  # Through center
    assert path_cost < center_path_cost, "Path should prefer lower-cost route"


# Test 4: No path exists (fully blocked)
def test_no_path_exists(blocked_grid):
    """Test that None is returned when no path exists."""
    start = (0, 0)
    end = (4, 4)
    
    path = blocked_grid.find_path(start, end)
    
    assert path is None, "No path should exist when completely blocked"


# Test 5: Start equals end
def test_start_equals_end(uniform_grid):
    """Test that start == end returns a single-element path."""
    position = (2, 2)
    
    path = uniform_grid.find_path(position, position)
    
    assert path is not None, "Path should exist for start == end"
    assert path == [position], "Path should be single element [start]"
    assert len(path) == 1, "Path should have exactly one element"


# Test 6: Invalid coordinates raise ValueError
def test_invalid_coordinates(uniform_grid):
    """Test that out-of-bounds coordinates raise ValueError."""
    # Test negative coordinates
    with pytest.raises(ValueError):
        uniform_grid.find_path((-1, 0), (0, 0))
    
    with pytest.raises(ValueError):
        uniform_grid.find_path((0, -1), (0, 0))
    
    # Test coordinates beyond grid size
    with pytest.raises(ValueError):
        uniform_grid.find_path((5, 0), (0, 0))
    
    with pytest.raises(ValueError):
        uniform_grid.find_path((0, 5), (0, 0))
    
    # Test both invalid
    with pytest.raises(ValueError):
        uniform_grid.find_path((10, 10), (0, 0))


# Additional test: Start or end is a wall
def test_start_or_end_is_wall(obstacle_grid):
    """Test that None is returned when start or end is a wall."""
    # Start is a wall
    path = obstacle_grid.find_path((1, 1), (0, 0))
    assert path is None, "No path when start is a wall"
    
    # End is a wall
    path = obstacle_grid.find_path((0, 0), (1, 1))
    assert path is None, "No path when end is a wall"
```

## Key Design Decisions

1. **Priority Queue with Tuple**: Using `(f_score, position)` tuples in the heap ensures proper ordering. The position comparison is a fallback that won't affect correctness since f_scores are unique for optimal paths.

2. **Closed Set Optimization**: Adding positions to `closed_set` when popped (not when pushed) prevents duplicate processing while allowing re-evaluation if a better path is found.

3. **Path Reconstruction**: Using a parent dictionary allows O(n) path reconstruction without storing full paths in the priority queue.

4. **Type Hints**: All methods include proper type hints for better IDE support and documentation.

5. **Test Coverage**: The 6 tests cover all required scenarios plus an additional test for wall detection at start/end positions.