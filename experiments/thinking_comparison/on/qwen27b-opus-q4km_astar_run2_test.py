from typing import List, Tuple, Optional
import heapq
from collections import deque

class AStarGrid:
    """A* pathfinding algorithm for weighted 2D grids."""
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D grid.
        
        Args:
            grid: 2D list where each value represents movement cost.
                  0 = impassable wall, positive int = cost to enter cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        # 4-directional movement: up, down, left, right
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    def _is_valid(self, row: int, col: int) -> bool:
        """Check if a coordinate is within bounds and not a wall."""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col] > 0
        return False
    
    def _heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Calculate Manhattan distance heuristic."""
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get all valid neighboring positions."""
        row, col = pos
        neighbors = []
        for dr, dc in self.directions:
            new_row, new_col = row + dr, col + dc
            if self._is_valid(new_row, new_col):
                neighbors.append((new_row, new_col))
        return neighbors
    
    def _reconstruct_path(
        self, 
        came_from: dict, 
        current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstruct the path from start to end."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def find_path(
        self, 
        start: Tuple[int, int], 
        end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting position as (row, col) tuple.
            end: Ending position as (row, col) tuple.
            
        Returns:
            List of (row, col) tuples representing the optimal path,
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate coordinates are within bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start position {start} is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End position {end} is out of bounds")
        
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None
        
        # Edge case: start equals end
        if start == end:
            return [start]
        
        # g_score: cost from start to current node
        g_score: dict[Tuple[int, int], int] = {start: 0}
        # f_score: g_score + heuristic
        f_score: dict[Tuple[int, int], int] = {start: self._heuristic(start, end)}
        
        # Open set: min-heap of (f_score, g_score, position)
        # Using g_score as secondary sort key for tie-breaking
        open_set: List[Tuple[int, int, Tuple[int, int]]] = [(f_score[start], 0, start)]
        open_set_hash: set[Tuple[int, int]] = {start}
        
        # Track where we came from for path reconstruction
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        
        while open_set:
            # Get node with lowest f_score
            _, _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)
            
            # Check if we reached the end
            if current == end:
                return self._reconstruct_path(came_from, current)
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                # Cost to move to neighbor (cost of entering the neighbor cell)
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], g_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        # No path found
        return None

import pytest
from typing import List, Tuple, Optional


def calculate_path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Calculate total cost of a path (sum of costs to enter each cell except start)."""
    if len(path) <= 1:
        return 0
    total_cost = 0
    for i in range(1, len(path)):
        row, col = path[i]
        total_cost += grid[row][col]
    return total_cost

def verify_path_validity(grid: List[List[int]], path: List[Tuple[int, int]]) -> bool:
    """Verify that a path is valid (all cells connected and not walls)."""
    if not path:
        return False
    
    # Check all cells are within bounds and not walls
    for row, col in path:
        if not (0 <= row < len(grid) and 0 <= col < len(grid[0])):
            return False
        if grid[row][col] == 0:
            return False
    
    # Check connectivity (each step is adjacent)
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return False
    
    return True

class TestAStarGrid:
    
    def test_simple_path_uniform_grid(self):
        """Test simple path on uniform grid with cost 1."""
        grid = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
        astar = AStarGrid(grid)
        
        start = (0, 0)
        end = (2, 3)
        
        path = astar.find_path(start, end)
        
        # Verify path exists
        assert path is not None, "Path should exist"
        
        # Verify path starts and ends correctly
        assert path[0] == start, f"Path should start at {start}"
        assert path[-1] == end, f"Path should end at {end}"
        
        # Verify path validity
        assert verify_path_validity(grid, path), "Path should be valid"
        
        # Verify optimality: Manhattan distance = 2 + 3 = 5 steps, cost = 5
        expected_cost = 5  # 5 cells entered (excluding start)
        actual_cost = calculate_path_cost(grid, path)
        assert actual_cost == expected_cost, f"Expected cost {expected_cost}, got {actual_cost}"
        
        # Path length should be 6 (start + 5 moves)
        assert len(path) == 6, f"Expected path length 6, got {len(path)}"
    
    def test_path_around_obstacles(self):
        """Test path finding around obstacles."""
        grid = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1],
        ]
        astar = AStarGrid(grid)
        
        start = (0, 0)
        end = (4, 4)
        
        path = astar.find_path(start, end)
        
        assert path is not None, "Path should exist around obstacles"
        assert path[0] == start
        assert path[-1] == end
        assert verify_path_validity(grid, path), "Path should be valid"
        
        # Verify no wall cells in path
        for row, col in path:
            assert grid[row][col] != 0, f"Path should not contain wall at ({row}, {col})"
    
    def test_weighted_grid_prefer_low_cost(self):
        """Test that path prefers lower-cost cells in weighted grid."""
        grid = [
            [1, 10, 10, 1],
            [1, 10, 10, 1],
            [1, 1, 1, 1],
        ]
        astar = AStarGrid(grid)
        
        start = (0, 0)
        end = (2, 3)
        
        path = astar.find_path(start, end)
        
        assert path is not None, "Path should exist"
        assert path[0] == start
        assert path[-1] == end
        assert verify_path_validity(grid, path), "Path should be valid"
        
        # Calculate cost of found path
        found_cost = calculate_path_cost(grid, path)
        
        # The optimal path goes: (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2) -> (2,3)
        # Cost = 1 + 1 + 1 + 1 + 1 = 5 (entering 5 cells with cost 1 each)
        # Going through top row would cost: 10 + 10 + 1 + 1 + 1 = 23
        assert found_cost == 5, f"Expected optimal cost 5, got {found_cost}"
        
        # Verify no high-cost cells (10) in path
        for row, col in path:
            assert grid[row][col] != 10, "Path should avoid high-cost cells"
    
    def test_no_path_exists_fully_blocked(self):
        """Test when no path exists due to complete blockage."""
        grid = [
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1],
        ]
        astar = AStarGrid(grid)
        
        start = (0, 0)
        end = (0, 2)
        
        path = astar.find_path(start, end)
        
        assert path is None, "No path should exist when fully blocked"
    
    def test_start_equals_end(self):
        """Test edge case where start equals end."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        astar = AStarGrid(grid)
        
        start = (1, 1)
        end = (1, 1)
        
        path = astar.find_path(start, end)
        
        assert path is not None, "Path should exist"
        assert path == [start], f"Expected path [{start}], got {path}"
        assert len(path) == 1, "Path should contain only start position"
    
    def test_invalid_coordinates(self):
        """Test that invalid coordinates raise ValueError."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        astar = AStarGrid(grid)
        
        # Test start out of bounds
        with pytest.raises(ValueError, match="Start position"):
            astar.find_path((-1, 0), (1, 1))
        
        with pytest.raises(ValueError, match="Start position"):
            astar.find_path((3, 0), (1, 1))
        
        with pytest.raises(ValueError, match="Start position"):
            astar.find_path((0, -1), (1, 1))
        
        with pytest.raises(ValueError, match="Start position"):
            astar.find_path((0, 3), (1, 1))
        
        # Test end out of bounds
        with pytest.raises(ValueError, match="End position"):
            astar.find_path((0, 0), (-1, 0))
        
        with pytest.raises(ValueError, match="End position"):
            astar.find_path((0, 0), (3, 0))
        
        with pytest.raises(ValueError, match="End position"):
            astar.find_path((0, 0), (0, -1))
        
        with pytest.raises(ValueError, match="End position"):
            astar.find_path((0, 0), (0, 3))
    
    def test_start_or_end_is_wall(self):
        """Test that start or end being a wall returns None."""
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
        astar = AStarGrid(grid)
        
        # Start is wall
        path = astar.find_path((1, 1), (0, 0))
        assert path is None, "Should return None when start is wall"
        
        # End is wall
        path = astar.find_path((0, 0), (1, 1))
        assert path is None, "Should return None when end is wall"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])