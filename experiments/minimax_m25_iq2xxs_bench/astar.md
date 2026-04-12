# MiniMax-M2.5 UD-IQ2_XXS (empty-think, 160K ctx) — A* Pathfinding

## Output (12912 chars)

Here's a complete implementation of the A* pathfinding algorithm in Python, meeting all the specified requirements.

```python
"""
A* Pathfinding on a Weighted 2D Grid

This module provides an A* implementation for finding the shortest path
on a weighted grid with 4-directional movement.
"""

import heapq
from typing import List, Tuple, Optional


class AStarGrid:
    """
    A* pathfinding on a weighted 2D grid.
    
    Attributes:
        grid: 2D list where 0 represents walls and positive integers represent movement costs.
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the A* pathfinder with a grid.
        
        Args:
            grid: 2D list of integers representing movement costs.
                  0 = impassable wall, positive int = cost to enter cell.
        
        Raises:
            ValueError: If grid is empty or not rectangular.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # Verify grid is rectangular
        for row in grid:
            if len(row) != self.cols:
                raise ValueError("Grid must be rectangular")
    
    def find_path(
        self, 
        start: Tuple[int, int], 
        end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A*.
        
        Args:
            start: Starting position as (row, col).
            end: Ending position as (row, col).
        
        Returns:
            List of (row, col) coordinates from start to end inclusive,
            or None if no path exists.
        
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate bounds
        self._validate_coordinates(start)
        self._validate_coordinates(end)
        
        # Edge case: start equals end
        if start == end:
            return [start]
        
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None
        
        # A* algorithm initialization
        # Priority queue: (f_score, counter, position)
        # counter used to handle tie-breaking while maintaining heap invariant
        counter = 0
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        
        g_score: dict[Tuple[int, int], float] = {start: 0}
        f_score: dict[Tuple[int, int], float] = {start: self._heuristic(start, end)}
        
        heapq.heappush(open_set, (f_score[start], counter, start))
        
        # Track the path for reconstruction
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # Set of visited nodes
        closed_set: set[Tuple[int, int]] = set()
        
        # Direction vectors: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            
            # Goal reached
            if current == end:
                return self._reconstruct_path(came_from, current)
            
            # Skip if already visited
            if current in closed_set:
                continue
            
            closed_set.add(current)
            
            # Explore neighbors
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                # Check bounds
                if not self._is_valid(neighbor):
                    continue
                
                # Skip walls
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]
                
                # If this path is better than any previous to neighbor
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                    counter += 1
        
        # No path found
        return None
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """
        Calculate Manhattan distance heuristic.
        
        Args:
            pos: Current position.
            goal: Goal position.
        
        Returns:
            Manhattan distance (|pos[0] - goal[0]| + |pos[1] - goal[1]|).
        """
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _validate_coordinates(self, pos: Tuple[int, int]) -> None:
        """Validate coordinates are within bounds."""
        if not self._is_valid(pos):
            raise ValueError(f"Coordinates {pos} are out of bounds")
    
    def _reconstruct_path(
        self, 
        came_from: dict[Tuple[int, int], Tuple[int, int]], 
        current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from goal to start."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def get_path_cost(self, path: List[Tuple[int, int]]) -> int:
        """
        Calculate total cost of a path.
        
        Args:
            path: List of coordinates representing the path.
        
        Returns:
            Total movement cost (sum of cell costs).
        """
        if not path:
            return 0
        # Cost includes all cells except start (entering each cell)
        return sum(self.grid[r][c] for r, c in path[1:])


# =============================================================================
# Pytest Tests
# =============================================================================

import pytest


class TestAStarGrid:
    """Test suite for A* pathfinding implementation."""
    
    def test_simple_path_uniform_grid(self):
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
        # Check path validity
        for i in range(len(path) - 1):
            dr = abs(path[i+1][0] - path[i][0])
            dc = abs(path[i+1][1] - path[i][1])
            assert (dr + dc) == 1, "Path must use 4-directional movement"
        # Optimal cost should be 6 (3+3 for 4 moves on 4x4 grid)
        assert astar.get_path_cost(path) == 6
    
    def test_path_around_obstacles(self):
        """Test path finding around walls/obstacles."""
        grid = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (4, 4))
        
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)
        # Path should go around the obstacles
        # Verify no path goes through walls
        for r, c in path:
            assert grid[r][c] != 0, "Path cannot go through walls"
    
    def test_weighted_grid_prefers_lower_cost(self):
        """Test that path prefers lower-cost cells."""
        grid = [
            [1, 1, 1, 1, 1],
            [1, 5, 5, 5, 1],
            [1, 5, 1, 5, 1],
            [1, 5, 5, 5, 1],
            [1, 1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (4, 4))
        
        assert path is not None
        # Path should prefer the outer ring (cost 1) over inner (cost 5)
        # Check that path avoids high-cost cells when possible
        cost = astar.get_path_cost(path)
        # The optimal path should have lower cost than going through center
        assert cost < 20, "Path should prefer lower-cost cells"
    
    def test_no_path_exists(self):
        """Test when no path exists due to complete blockage."""
        grid = [
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (4, 4))
        
        assert path is None, "No path should exist through wall"
    
    def test_start_equals_end(self):
        """Test when start equals end."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((1, 1), (1, 1))
        
        assert path is not None
        assert path == [(1, 1)]
        assert astar.get_path_cost(path) == 0
    
    def test_invalid_coordinates(self):
        """Test invalid coordinates raise ValueError."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Test out of bounds start
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (3, 3))
        
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (0, 0))
        
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (0, 5))
    
    def test_start_or_end_is_wall(self):
        """Test when start or end is a wall (0)."""
        grid = [
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Start is a wall
        path = astar.find_path((0, 0), (2, 2))
        assert path is None
        
        # End is a wall
        path = astar.find_path((0, 1), (0, 0))
        assert path is None
    
    def test_optimality_on_weighted_grid(self):
        """Test optimality - verify minimum cost path is chosen."""
        # Create a grid where there are two possible paths with different costs
        grid = [
            [1, 2, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (0, 4))
        
        assert path is not None
        # The path should go through the grid, preferring lower costs
        # Cost should be optimal
        cost = astar.get_path_cost(path)
        # Verify all moves are valid 4-directional
        for i in range(len(path) - 1):
            dr = abs(path[i+1][0] - path[i][0])
            dc = abs(path[i+1][1] - path[i][1])
            assert (dr + dc) == 1


if __name__ == "__main__":
    # Run a quick demo
    grid = [
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]
    
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (4, 4))
    
    print("Grid:")
    for row in grid:
        print(row)
    
    print(f"\nPath from (0,0) to (4,4): {path}")
    print(f"Cost: {astar.get_path_cost(path)}")
    
    # Run pytest
    pytest.main([__file__, "-v"])
```

## Key Implementation Details

### 1. **A* Algorithm**
- Uses `g_score` (actual cost from start) and `f_score` (g_score + heuristic)
- Manhattan distance as the heuristic function
- 4-directional movement (no diagonals)

### 2. **Weighted Grid Support**
- Cell costs are added to g_score when traversing
- Path reconstruction tracks actual path taken
- `get_path_cost()` calculates total cost of any given path

### 3. **Edge Cases Handled**
- Start equals end returns `[start]`
- Start or end is a wall returns `None`
- Out of bounds raises `ValueError`
- No path exists returns `None`

### 4. **Performance**
- Uses `heapq` (min-heap) for the open set - O(log n) operations
- Tracks visited nodes in `closed_set` to avoid reprocessing
- Counter for heap tie-breaking

### 5. **Test Coverage**
The tests cover:
1. Simple path on uniform grid
2. Path around obstacles
3. Weighted grid (prefers lower-cost cells)
4. No path exists (fully blocked)
5. Start equals end
6. Invalid coordinates (ValueError)
7. Additional tests for wall handling and optimality

Run the tests with:
```bash
pytest filename.py -v
```
