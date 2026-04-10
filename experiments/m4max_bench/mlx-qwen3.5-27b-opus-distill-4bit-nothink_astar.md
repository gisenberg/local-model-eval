# Qwen 3.5 27B Opus-Distilled 4bit MLX — astar

## Thinking (542 chars)

```
Let me think about how to approach this coding problem.

Problem: Implement A* pathfinding on a weighted 2D grid in Python. Requirements:

1. Class AStarGrid with __init__(self, grid: List[List[int]]) where grid values represent movement cost (0 = impassable wall, positive int = cost to enter that cell)
2. find_path(start, end) -> Optional[List[Tuple[int, int]]] — shortest path or None
3. 4-directional movement, Manhattan heuristic, heapq open set
4. Handle: start==end, walls, out-of-bounds (ValueError)

Let me implement this solution.

```

## Output



```python
"""
A* Pathfinding Implementation for Weighted 2D Grids

This module provides an A* pathfinding algorithm implementation for finding
optimal paths through weighted 2D grids with obstacle avoidance.
"""

from __future__ import annotations

import heapq
from typing import List, Optional, Tuple, Dict, Set

__all__ = ["AStarGrid"]


class AStarGrid:
    """
    A* pathfinding implementation for weighted 2D grids.
    
    Supports 4-directional movement (up, down, left, right) with
    Manhattan distance heuristic for optimal pathfinding.
    
    Attributes:
        grid: 2D list where 0 represents walls and positive integers
              represent movement costs to enter cells.
    
    Example:
        >>> grid = [
        ...     [1, 1, 1, 0, 1],
        ...     [1, 1, 1, 0, 1],
        ...     [1, 0, 0, 1, 1],
        ...     [1, 1, 1, 1, 1]
        ... ]
        >>> astar = AStarGrid(grid)
        >>> path = astar.find_path((0, 0), (3, 4))
    """
    
    # Direction vectors for 4-directional movement: up, down, left, right
    DIRECTIONS: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the A* grid with movement costs.
        
        Args:
            grid: 2D list where each value represents the cost to enter
                  that cell. 0 = impassable wall, positive int = cost.
        
        Raises:
            ValueError: If grid is empty, has inconsistent row lengths,
                       or contains invalid values.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty")
        
        # Validate grid structure
        row_length = len(grid[0])
        for row in grid:
            if len(row) != row_length:
                raise ValueError("All rows must have the same length")
            for val in row:
                if val < 0:
                    raise ValueError("Grid values must be non-negative")
        
        self._grid = [row[:] for row in grid]  # Deep copy
        self._rows = len(grid)
        self._cols = row_length
    
    def _validate_position(self, pos: Tuple[int, int]) -> None:
        """
        Validate that a position is within grid bounds.
        
        Args:
            pos: (row, col) tuple to validate.
        
        Raises:
            ValueError: If position is out of bounds.
        """
        row, col = pos
        if not (0 <= row < self._rows and 0 <= col < self._cols):
            raise ValueError(
                f"Position {pos} is out of bounds "
                f"(grid size: {self._rows}x{self._cols})"
            )
    
    def _manhattan_heuristic(self, pos: Tuple[int, int], 
                             goal: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance heuristic between two positions.
        
        Args:
            pos: Current position (row, col).
            goal: Goal position (row, col).
        
        Returns:
            Manhattan distance (sum of absolute coordinate differences).
        """
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], int]]:
        """
        Get valid neighboring positions and their movement costs.
        
        Args:
            pos: Current position (row, col).
        
        Returns:
            List of (neighbor_pos, cost) tuples for valid neighbors.
        """
        neighbors = []
        row, col = pos
        
        for dr, dc in self.DIRECTIONS:
            new_row, new_col = row + dr, col + dc
            
            # Check bounds
            if not (0 <= new_row < self._rows and 0 <= new_col < self._cols):
                continue
            
            # Check if passable (cost > 0)
            cost = self._grid[new_row][new_col]
            if cost > 0:
                neighbors.append(((new_row, new_col), cost))
        
        return neighbors
    
    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]],
                          current: Tuple[int, int],
                          start: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from start to current by backtracking.
        
        Args:
            came_from: Dictionary mapping each position to its predecessor.
            current: End position of the path.
            start: Start position of the path.
        
        Returns:
            List of positions from start to current.
        """
        path = [current]
        while current != start:
            current = came_from[current]
            path.append(current)
        return list(reversed(path))
    
    def find_path(self, start: Tuple[int, int], 
                  end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: Starting position as (row, col) tuple.
            end: Ending position as (row, col) tuple.
        
        Returns:
            List of (row, col) tuples representing the optimal path,
            or None if no path exists.
        
        Raises:
            ValueError: If start or end positions are out of bounds,
                       or if start/end cells are walls (cost = 0).
        
        Algorithm:
            Uses A* with:
            - f(n) = g(n) + h(n)
            - g(n) = actual cost from start to n
            - h(n) = Manhattan distance heuristic to goal
        """
        # Validate positions
        self._validate_position(start)
        self._validate_position(end)
        
        # Check if start or end are walls
        if self._grid[start[0]][start[1]] == 0:
            raise ValueError(f"Start position {start} is a wall")
        if self._grid[end[0]][end[1]] == 0:
            raise ValueError(f"End position {end} is a wall")
        
        # Handle start == end case
        if start == end:
            return [start]
        
        # Priority queue: (f_score, counter, position)
        # Counter breaks ties consistently
        counter = 0
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        
        # g_score: cost from start to each node
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        
        # f_score: g_score + heuristic
        f_score: Dict[Tuple[int, int], int] = {
            start: self._manhattan_heuristic(start, end)
        }
        
        # Track predecessors for path reconstruction
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # Add start to open set
        heapq.heappush(open_set, (f_score[start], counter, start))
        
        # Set of positions in open set for O(1) lookup
        open_set_set: Set[Tuple[int, int]] = {start}
        
        while open_set:
            # Get node with lowest f_score
            _, _, current = heapq.heappop(open_set)
            open_set_set.discard(current)
            
            # Found goal
            if current == end:
                return self._reconstruct_path(came_from, current, start)
            
            # Explore neighbors
            for neighbor, move_cost in self._get_neighbors(current):
                tentative_g = g_score[current] + move_cost
                
                # Found better path to neighbor
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._manhattan_heuristic(neighbor, end)
                    f_score[neighbor] = f
                    
                    if neighbor not in open_set_set:
                        counter += 1
                        heapq.heappush(open_set, (f, counter, neighbor))
                        open_set_set.add(neighbor)
        
        # No path found
        return None
```

---

```python
"""
Pytest test suite for A* pathfinding implementation.
"""

import pytest
from typing import List, Tuple, Optional
from a_star_grid import AStarGrid


class TestAStarGrid:
    """Test suite for AStarGrid class."""
    
    def test_simple_path(self) -> None:
        """Test basic pathfinding on a simple grid."""
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
        assert len(path) == 5  # Manhattan distance + 1
    
    def test_path_with_walls(self) -> None:
        """Test pathfinding around obstacles."""
        grid = [
            [1, 1, 1, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (3, 4))
        
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (3, 4)
        # Verify path doesn't go through walls
        for pos in path:
            assert grid[pos[0]][pos[1]] != 0
    
    def test_no_path_exists(self) -> None:
        """Test that None is returned when no path exists."""
        grid = [
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (0, 2))
        
        assert path is None
    
    def test_start_equals_end(self) -> None:
        """Test that start == end returns a single-element path."""
        grid = [[1, 1], [1, 1]]
        astar = AStarGrid(grid)
        path = astar.find_path((1, 1), (1, 1))
        
        assert path == [(1, 1)]
    
    def test_weighted_optimal_path(self) -> None:
        """Test that algorithm finds minimum cost path with varying weights."""
        grid = [
            [1, 10, 1],
            [1, 10, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is not None
        # Optimal path should go around the expensive center
        # Path: (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2) = cost 4
        # Not through center which would cost 1 + 10 + 1 = 12 minimum
        total_cost = sum(grid[r][c] for r, c in path[1:])
        assert total_cost == 4
    
    def test_validation_errors(self) -> None:
        """Test proper error handling for invalid inputs."""
        grid = [[1, 1], [1, 1]]
        astar = AStarGrid(grid)
        
        # Out of bounds start
        with pytest.raises(ValueError, match="out of bounds"):
            astar.find_path((-1, 0), (1, 1))
        
        # Out of bounds end
        with pytest.raises(ValueError, match="out of bounds"):
            astar.find_path((0, 0), (5, 5))
        
        # Start is wall
        grid_wall = [[0, 1], [1, 1]]
        astar_wall = AStarGrid(grid_wall)
        with pytest.raises(ValueError, match="Start position"):
            astar_wall.find_path((0, 0), (1, 1))
        
        # End is wall
        with pytest.raises(ValueError, match="End position"):
            astar_wall.find_path((1, 1), (0, 0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

## Summary

### Implementation Details

| Component | Description |
|-----------|-------------|
| **Heuristic** | Manhattan distance (admissible for 4-directional movement) |
| **Open Set** | Python `heapq` priority queue with `(f_score, counter, position)` |
| **Tie-Breaking** | Monotonic counter ensures consistent ordering |
| **Path Reconstruction** | Backtracking via `came_from` dictionary |

### Key Features

1. **Optimal Paths**: A* with admissible heuristic guarantees optimality
2. **Weighted Grids**: Supports arbitrary positive movement costs
3. **Input Validation**: Comprehensive bounds and wall checking
4. **Edge Cases**: Handles start==end, unreachable goals, blocked paths

### Test Coverage

| Test | Purpose |
|------|---------|
| `test_simple_path` | Basic functionality on open grid |
| `test_path_with_walls` | Navigation around obstacles |
| `test_no_path_exists` | Returns None when blocked |
| `test_start_equals_end` | Single-cell path edge case |
| `test_weighted_optimal_path` | Minimum cost with varying weights |
| `test_validation_errors` | Proper exception handling |
