from typing import List, Tuple, Optional
import heapq


class AStarGrid:
    """A* pathfinding on a weighted 2D grid with 4-directional movement."""
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid for pathfinding.
        
        Args:
            grid: 2D list where each value represents movement cost.
                  0 = impassable wall, positive int = cost to enter cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        self._directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if position is an impassable wall (cost = 0)."""
        row, col = pos
        return self.grid[row][col] == 0
    
    def _get_cost(self, pos: Tuple[int, int]) -> int:
        """Get the cost to enter a cell."""
        return self.grid[pos[0]][pos[1]]
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance heuristic.
        
        Manhattan distance is admissible for 4-directional movement
        as it never overestimates the actual cost.
        """
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], int]]:
        """
        Get valid neighboring positions and their entry costs.
        
        Returns list of (neighbor_position, entry_cost) tuples.
        """
        neighbors = []
        row, col = pos
        
        for dr, dc in self._directions:
            new_row, new_col = row + dr, col + dc
            new_pos = (new_row, new_col)
            
            if self._is_valid(new_pos) and not self._is_wall(new_pos):
                neighbors.append((new_pos, self._get_cost(new_pos)))
        
        return neighbors
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting position as (row, col) tuple.
            end: Ending position as (row, col) tuple.
            
        Returns:
            List of (row, col) coordinates from start to end inclusive,
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate coordinates are within bounds
        if not self._is_valid(start):
            raise ValueError(f"Start position {start} is out of bounds")
        if not self._is_valid(end):
            raise ValueError(f"End position {end} is out of bounds")
        
        # Handle edge case: start equals end
        if start == end:
            return [start]
        
        # Check if start or end is a wall
        if self._is_wall(start) or self._is_wall(end):
            return None
        
        # Open set: min-heap of (f_score, g_score, position)
        # Using g_score as secondary sort key for stability
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))
        
        # Track g_scores (cost from start to each position)
        g_scores: dict[Tuple[int, int], int] = {start: 0}
        
        # Track parent for path reconstruction
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # Closed set: visited positions
        closed_set: set[Tuple[int, int]] = set()
        
        while open_set:
            # Get position with lowest f_score
            _, current_g, current = heapq.heappop(open_set)
            
            # Skip if already processed
            if current in closed_set:
                continue
            
            # Check if we reached the goal
            if current == end:
                return self._reconstruct_path(came_from, start, end)
            
            # Add to closed set
            closed_set.add(current)
            
            # Explore neighbors
            for neighbor, move_cost in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g = current_g + move_cost
                
                # Update if we found a better path
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    came_from[neighbor] = current
        
        # No path found
        return None
    
    def _reconstruct_path(
        self, 
        came_from: dict[Tuple[int, int], Tuple[int, int]], 
        start: Tuple[int, int], 
        end: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from end to start using parent pointers."""
        path = [end]
        current = end
        
        while current != start:
            if current not in came_from:
                return []  # Should not happen if algorithm is correct
            current = came_from[current]
            path.append(current)
        
        return list(reversed(path))
    
    def calculate_path_cost(self, path: List[Tuple[int, int]]) -> int:
        """
        Calculate the total cost of a path.
        
        Args:
            path: List of positions in the path.
            
        Returns:
            Total cost to traverse the path (sum of entry costs).
        """
        if not path:
            return 0
        
        # Cost is sum of entry costs for all cells except start
        total_cost = 0
        for pos in path[1:]:  # Skip start position
            total_cost += self._get_cost(pos)
        
        return total_cost

import pytest
from typing import List, Tuple, Optional
import sys
sys.path.insert(0, '.')




class TestAStarGrid:
    """Test suite for AStarGrid pathfinding."""
    
    def test_simple_path_uniform_grid(self):
        """Test simple path on uniform grid with equal costs."""
        grid = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
        astar = AStarGrid(grid)
        
        start = (0, 0)
        end = (2, 3)
        
        path = astar.find_path(start, end)
        
        # Verify path exists and is valid
        assert path is not None
        assert path[0] == start
        assert path[-1] == end
        assert len(path) == 6  # Manhattan distance + 1
        
        # Verify optimality: minimum cost is 5 (5 moves, each costing 1)
        cost = astar.calculate_path_cost(path)
        assert cost == 5
        
        # Verify each step is valid (adjacent cells)
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            assert abs(r1 - r2) + abs(c1 - c2) == 1  # Adjacent cells
    
    def test_path_around_obstacles(self):
        """Test pathfinding around obstacles (walls)."""
        grid = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],  # Wall row
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],  # Partial wall
            [1, 1, 1, 1, 1],
        ]
        astar = AStarGrid(grid)
        
        start = (0, 0)
        end = (4, 4)
        
        path = astar.find_path(start, end)
        
        # Verify path exists
        assert path is not None
        assert path[0] == start
        assert path[-1] == end
        
        # Verify no wall cells in path
        for pos in path:
            assert grid[pos[0]][pos[1]] != 0
    
    def test_weighted_grid_prefers_lower_cost(self):
        """Test that path prefers lower-cost cells over shorter distance."""
        # Create a grid where taking a longer path with lower costs is optimal
        grid = [
            [1, 100, 100, 1],
            [1, 1, 1, 1],
            [1, 100, 100, 1],
        ]
        astar = AStarGrid(grid)
        
        start = (0, 0)
        end = (0, 3)
        
        path = astar.find_path(start, end)
        
        # Verify path exists
        assert path is not None
        assert path[0] == start
        assert path[-1] == end
        
        # Direct path cost: 100 + 100 + 1 = 201
        # Detour path: 1 + 1 + 1 + 1 + 1 + 1 = 6 (go down, across, up)
        cost = astar.calculate_path_cost(path)
        
        # The algorithm should find the detour path with cost 6
        assert cost == 6
        
        # Verify the path goes through row 1 (the cheap row)
        assert any(pos[0] == 1 for pos in path)
    
    def test_no_path_exists_fully_blocked(self):
        """Test that None is returned when no path exists."""
        grid = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1],
        ]
        astar = AStarGrid(grid)
        
        start = (0, 0)
        end = (2, 2)
        
        path = astar.find_path(start, end)
        
        # No path should exist
        assert path is None
    
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
        
        # Should return path with just the start position
        assert path == [start]
        assert len(path) == 1
    
    def test_invalid_coordinates(self):
        """Test that ValueError is raised for invalid coordinates."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        astar = AStarGrid(grid)
        
        # Test out of bounds start
        with pytest.raises(ValueError, match="Start position"):
            astar.find_path((-1, 0), (1, 1))
        
        with pytest.raises(ValueError, match="Start position"):
            astar.find_path((3, 0), (1, 1))
        
        # Test out of bounds end
        with pytest.raises(ValueError, match="End position"):
            astar.find_path((0, 0), (0, 3))
        
        with pytest.raises(ValueError, match="End position"):
            astar.find_path((0, 0), (-1, -1))
    
    def test_start_or_end_is_wall(self):
        """Test that None is returned when start or end is a wall."""
        grid = [
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 0],
        ]
        astar = AStarGrid(grid)
        
        # Start is wall
        path = astar.find_path((0, 0), (2, 2))
        assert path is None
        
        # End is wall
        path = astar.find_path((0, 1), (2, 2))
        assert path is None
    
    def test_path_optimality_verification(self):
        """Verify that the returned path is truly optimal."""
        grid = [
            [1, 2, 1, 1],
            [1, 1, 1, 1],
            [1, 2, 2, 1],
        ]
        astar = AStarGrid(grid)
        
        start = (0, 0)
        end = (2, 3)
        
        path = astar.find_path(start, end)
        assert path is not None
        
        # Calculate cost of found path
        found_cost = astar.calculate_path_cost(path)
        
        # Brute force verification: check all possible paths
        def get_all_paths(grid_obj, start, end, max_depth=20):
            """Get all possible paths with their costs (for small grids only)."""
            from collections import deque
            
            results = []
            queue = deque([(start, [start], 0)])
            
            while queue:
                current, path, cost = queue.popleft()
                
                if current == end:
                    results.append((path, cost))
                    continue
                
                if len(path) > max_depth:
                    continue
                
                for neighbor, move_cost in grid_obj._get_neighbors(current):
                    if neighbor not in path:  # Avoid cycles
                        new_path = path + [neighbor]
                        new_cost = cost + move_cost
                        queue.append((neighbor, new_path, new_cost))
            
            return results
        
        all_paths = get_all_paths(astar, start, end)
        assert len(all_paths) > 0
        
        # Find minimum cost among all paths
        min_cost = min(cost for _, cost in all_paths)
        
        # Verify our path has minimum cost
        assert found_cost == min_cost, f"Found cost {found_cost} != minimum {min_cost}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])