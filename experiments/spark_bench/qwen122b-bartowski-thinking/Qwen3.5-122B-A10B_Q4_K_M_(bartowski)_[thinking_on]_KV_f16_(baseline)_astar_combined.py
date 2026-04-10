import heapq
import math
from typing import List, Tuple, Optional, Dict, Set

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* Grid with a 2D list of movement costs.
        
        Args:
            grid: 2D list where 0 represents a wall and positive integers 
                  represent the cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, r: int, c: int) -> bool:
        """Check if a cell is a wall (cost 0)."""
        return self.grid[r][c] == 0

    def _heuristic(self, r1: int, c1: int, r2: int, c2: int) -> int:
        """
        Calculate Manhattan distance between two points.
        Used as the admissible heuristic for A*.
        """
        return abs(r1 - r2) + abs(c1 - c2)

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Tuple (row, col) representing the starting position.
            end: Tuple (row, col) representing the target position.
            
        Returns:
            List of (row, col) tuples representing the path from start to end,
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate bounds
        if not self._is_valid(*start) or not self._is_valid(*end):
            raise ValueError("Start or end coordinates are out of bounds.")
        
        # Handle start == end
        if start == end:
            return [start]
        
        # Check if start or end is a wall
        if self._is_wall(*start) or self._is_wall(*end):
            return None

        # Directions: Up, Down, Left, Right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Priority Queue: (f_score, row, col)
        # We use a counter implicitly via tuple comparison (f_score, row, col)
        open_set: List[Tuple[int, int, int]] = []
        
        # g_score: Cost from start to current node
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        
        # f_score: g_score + heuristic
        f_score: Dict[Tuple[int, int], int] = {start: self._heuristic(*start, *end)}
        
        # Predecessors for path reconstruction
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # Push start node
        heapq.heappush(open_set, (f_score[start], start[0], start[1]))
        
        # Closed set to avoid re-processing nodes
        closed_set: Set[Tuple[int, int]] = set()
        
        while open_set:
            # Pop node with lowest f_score
            _, current_r, current_c = heapq.heappop(open_set)
            current = (current_r, current_c)
            
            # If we reached the end
            if current == end:
                return self._reconstruct_path(came_from, current)
            
            # Skip if already processed
            if current in closed_set:
                continue
            closed_set.add(current)
            
            # Explore neighbors
            for dr, dc in directions:
                neighbor_r, neighbor_c = current_r + dr, current_c + dc
                neighbor = (neighbor_r, neighbor_c)
                
                # Check bounds and walls
                if not self._is_valid(neighbor_r, neighbor_c):
                    continue
                if self._is_wall(neighbor_r, neighbor_c):
                    continue
                if neighbor in closed_set:
                    continue
                
                # Cost to enter the neighbor cell
                move_cost = self.grid[neighbor_r][neighbor_c]
                tentative_g = g_score[current] + move_cost
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor_r, neighbor_c, *end)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor_r, neighbor_c))
        
        # No path found
        return None

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from end to start using the came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


# =============================================================================
# Pytest Tests
# =============================================================================

import pytest

def calculate_path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Helper to calculate total cost of a path (sum of costs of cells entered)."""
    if not path:
        return 0
    # Cost is sum of all cells in path EXCEPT the start cell (since we don't 'enter' start)
    # Based on "cost to enter that cell" logic.
    total_cost = 0
    for i in range(1, len(path)):
        r, c = path[i]
        total_cost += grid[r][c]
    return total_cost

def test_simple_path_uniform_grid():
    """Test 1: Simple path on uniform grid."""
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
    # Optimal path length is 5 cells (Manhattan distance + 1). 
    # Cost = 4 cells entered * 1 cost = 4.
    cost = calculate_path_cost(grid, path)
    assert cost == 4
    assert len(path) == 5

def test_path_around_obstacles():
    """Test 2: Path around obstacles."""
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 3))
    
    assert path is not None
    # Path must go around the wall in row 1
    # One valid path: (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3)
    # Cost: 1+1+1+1+1 = 5
    cost = calculate_path_cost(grid, path)
    assert cost == 5
    # Verify no wall cells are in path
    for r, c in path:
        assert grid[r][c] != 0

def test_weighted_grid_optimality():
    """Test 3: Weighted grid (path prefers lower-cost cells)."""
    grid = [
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Start (0,0), End (2,2)
    # Path 1 (Diagonal-ish): (0,0)->(0,1)->(0,2)->(1,2)->(2,2) Cost: 1+1+1+1 = 4
    # Path 2 (Through middle): (0,0)->(1,0)->(1,1)->(1,2)->(2,2) Cost: 1+10+1+1 = 13
    # Path 3 (Bottom): (0,0)->(1,0)->(2,0)->(2,1)->(2,2) Cost: 1+1+1+1 = 4
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    cost = calculate_path_cost(grid, path)
    # Optimal cost should be 4 (avoiding the 10 cost cell)
    assert cost == 4
    # Ensure the expensive cell is not in the path
    assert (1, 1) not in path

def test_no_path_exists():
    """Test 4: No path exists (fully blocked)."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is None

def test_start_equals_end():
    """Test 5: Start equals end."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    
    assert path is not None
    assert path == [(0, 0)]
    assert calculate_path_cost(grid, path) == 0

def test_invalid_coordinates():
    """Test 6: Invalid coordinates (out of bounds)."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Test start out of bounds
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
        
    # Test end out of bounds
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))

def test_start_or_end_is_wall():
    """Additional Test: Start or end is a wall."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Start is wall
    path = astar.find_path((0, 0), (1, 1))
    assert path is None
    
    # End is wall
    path = astar.find_path((1, 0), (0, 0))
    assert path is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
