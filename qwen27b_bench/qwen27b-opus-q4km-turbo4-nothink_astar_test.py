from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    """A* pathfinding algorithm on a weighted 2D grid."""
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid with movement costs.
        
        Args:
            grid: 2D list where each value represents the cost to enter that cell.
                  0 = impassable wall, positive int = cost to enter.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        self._validate_grid()
    
    def _validate_grid(self) -> None:
        """Validate that the grid is well-formed."""
        if self.rows == 0:
            return
        for row in self.grid:
            if len(row) != self.cols:
                raise ValueError("Grid rows must have equal length")
            for cell in row:
                if cell < 0:
                    raise ValueError("Grid values must be non-negative")
    
    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within bounds."""
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _is_passable(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is passable (not a wall)."""
        if not self._is_valid(pos):
            return False
        return self.grid[pos[0]][pos[1]] > 0
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid 4-directional neighbors of a position."""
        row, col = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        neighbors = []
        for dr, dc in directions:
            new_pos = (row + dr, col + dc)
            if self._is_passable(new_pos):
                neighbors.append(new_pos)
        return neighbors
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def _get_cost(self, pos: Tuple[int, int]) -> int:
        """Get the cost to enter a cell."""
        return self.grid[pos[0]][pos[1]]
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting position as (row, col) tuple
            end: Ending position as (row, col) tuple
            
        Returns:
            List of (row, col) coordinates from start to end inclusive,
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate coordinates are within bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start position {start} is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End position {end} is out of bounds")
        
        # Edge case: start equals end
        if start == end:
            return [start]
        
        # Edge case: start or end is a wall
        if not self._is_passable(start) or not self._is_passable(end):
            return None
        
        # Open set: min-heap of (f_score, -g_score, position)
        # Using -g_score for tie-breaking (prefer lower g_score)
        open_set: List[Tuple[int, int, Tuple[int, int]]] = [(0, 0, start)]
        
        # Track g_score (actual cost from start) for each position
        g_score: dict[Tuple[int, int], int] = {start: 0}
        
        # Track where we came from for path reconstruction
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # Closed set: visited positions
        closed_set: set[Tuple[int, int]] = set()
        
        while open_set:
            # Get node with lowest f_score
            _, _, current = heapq.heappop(open_set)
            
            # If we've already processed this node with a better path, skip
            if current in closed_set:
                continue
            
            # Check if we reached the goal
            if current == end:
                return self._reconstruct_path(came_from, start, end)
            
            # Add to closed set
            closed_set.add(current)
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_score[current] + self._get_cost(neighbor)
                
                # If this path to neighbor is better, update it
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (f_score, -tentative_g, neighbor))
        
        # No path found
        return None
    
    def _reconstruct_path(self, came_from: dict[Tuple[int, int], Tuple[int, int]], 
                          start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from end to start."""
        path = [end]
        current = end
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

import pytest
from typing import List, Tuple, Optional
import sys
sys.path.insert(0, '.')


def calculate_path_cost(grid: AStarGrid, path: List[Tuple[int, int]]) -> int:
    """Calculate the total cost of a path (excluding start cell)."""
    if len(path) <= 1:
        return 0
    cost = 0
    for pos in path[1:]:
        cost += grid.grid[pos[0]][pos[1]]
    return cost

def verify_path_validity(grid: AStarGrid, path: List[Tuple[int, int]], 
                         start: Tuple[int, int], end: Tuple[int, int]) -> bool:
    """Verify that a path is valid (connects start to end, all cells passable, adjacent cells)."""
    if not path:
        return False
    if path[0] != start or path[-1] != end:
        return False
    for i in range(len(path) - 1):
        curr = path[i]
        next_pos = path[i + 1]
        # Check adjacency (4-directional)
        if abs(curr[0] - next_pos[0]) + abs(curr[1] - next_pos[1]) != 1:
            return False
    return True

class TestAStarGrid:
    
    def test_simple_path_uniform_grid(self):
        """Test simple path on a uniform grid with cost 1."""
        grid_data = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
        grid = AStarGrid(grid_data)
        start = (0, 0)
        end = (2, 3)
        
        path = grid.find_path(start, end)
        
        assert path is not None, "Path should exist"
        assert verify_path_validity(grid, path, start, end), "Path should be valid"
        assert path[0] == start and path[-1] == end, "Path should start and end correctly"
        # Minimum cost: 5 steps (2 down + 3 right) = 5
        assert calculate_path_cost(grid, path) == 5, f"Expected cost 5, got {calculate_path_cost(grid, path)}"
    
    def test_path_around_obstacles(self):
        """Test path finding around obstacles (walls)."""
        grid_data = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
        grid = AStarGrid(grid_data)
        start = (0, 0)
        end = (2, 4)
        
        path = grid.find_path(start, end)
        
        assert path is not None, "Path should exist"
        assert verify_path_validity(grid, path, start, end), "Path should be valid"
        # Must go around: down, right, right, right, up, right, right = 7 steps
        assert calculate_path_cost(grid, path) == 7, f"Expected cost 7, got {calculate_path_cost(grid, path)}"
    
    def test_weighted_grid_optimal_path(self):
        """Test that path prefers lower-cost cells on weighted grid."""
        grid_data = [
            [1, 10, 10, 1],
            [1, 1, 1, 1],
            [1, 10, 10, 1],
        ]
        grid = AStarGrid(grid_data)
        start = (0, 0)
        end = (2, 3)
        
        path = grid.find_path(start, end)
        
        assert path is not None, "Path should exist"
        assert verify_path_validity(grid, path, start, end), "Path should be valid"
        # Optimal path: (0,0)->(1,0)->(1,1)->(1,2)->(1,3)->(2,3) = 5 cells, cost = 5
        # Alternative through top: (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3) = 5 cells, cost = 10+10+1+1+1 = 23
        cost = calculate_path_cost(grid, path)
        assert cost == 5, f"Expected optimal cost 5, got {cost}"
        # Verify path doesn't use expensive cells
        for pos in path[1:]:
            assert grid.grid[pos[0]][pos[1]] == 1, f"Path should avoid expensive cells, but used {pos} with cost {grid.grid[pos[0]][pos[1]]}"
    
    def test_no_path_exists(self):
        """Test when no path exists (fully blocked)."""
        grid_data = [
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
        ]
        grid = AStarGrid(grid_data)
        start = (0, 0)
        end = (0, 2)
        
        path = grid.find_path(start, end)
        
        assert path is None, "No path should exist"
    
    def test_start_equals_end(self):
        """Test when start position equals end position."""
        grid_data = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        grid = AStarGrid(grid_data)
        start = (1, 1)
        end = (1, 1)
        
        path = grid.find_path(start, end)
        
        assert path == [start], f"Expected [{start}], got {path}"
        assert calculate_path_cost(grid, path) == 0, "Cost should be 0 for start==end"
    
    def test_invalid_coordinates(self):
        """Test that invalid coordinates raise ValueError."""
        grid_data = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        grid = AStarGrid(grid_data)
        
        # Test invalid start
        with pytest.raises(ValueError, match="Start position"):
            grid.find_path((-1, 0), (1, 1))
        
        with pytest.raises(ValueError, match="Start position"):
            grid.find_path((3, 0), (1, 1))
        
        # Test invalid end
        with pytest.raises(ValueError, match="End position"):
            grid.find_path((1, 1), (0, -1))
        
        with pytest.raises(ValueError, match="End position"):
            grid.find_path((1, 1), (2, 3))
    
    def test_start_or_end_is_wall(self):
        """Test when start or end is a wall."""
        grid_data = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
        grid = AStarGrid(grid_data)
        
        # Start is wall
        path = grid.find_path((1, 1), (0, 0))
        assert path is None, "Should return None when start is wall"
        
        # End is wall
        path = grid.find_path((0, 0), (1, 1))
        assert path is None, "Should return None when end is wall"
    
    def test_complex_weighted_grid(self):
        """Test optimal path on a more complex weighted grid."""
        grid_data = [
            [1, 2, 1, 100, 1],
            [1, 1, 1, 100, 1],
            [1, 100, 100, 100, 1],
            [1, 1, 1, 1, 1],
        ]
        grid = AStarGrid(grid_data)
        start = (0, 0)
        end = (3, 4)
        
        path = grid.find_path(start, end)
        
        assert path is not None, "Path should exist"
        assert verify_path_validity(grid, path, start, end), "Path should be valid"
        
        # Calculate expected optimal cost
        # Path: (0,0)->(1,0)->(2,0)->(3,0)->(3,1)->(3,2)->(3,3)->(3,4)
        # Cost: 1+1+1+1+1+1+1 = 7
        cost = calculate_path_cost(grid, path)
        assert cost == 7, f"Expected optimal cost 7, got {cost}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])