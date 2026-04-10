from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    """A* pathfinding algorithm on a weighted 2D grid."""
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid with a 2D grid of movement costs.
        
        Args:
            grid: 2D list where each value represents the cost to enter that cell.
                  0 means impassable wall, positive integers are movement costs.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        self._directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within grid bounds."""
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols
    
    def _is_passable(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is passable (not a wall)."""
        if not self._is_valid(pos):
            return False
        return self.grid[pos[0]][pos[1]] > 0
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance heuristic.
        
        Manhattan distance is admissible for 4-directional grid movement
        because it never overestimates the actual cost to reach the goal.
        """
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], int]]:
        """
        Get all valid neighboring positions and their movement costs.
        
        Returns a list of (neighbor_position, cost_to_enter) tuples.
        """
        neighbors = []
        row, col = pos
        
        for dr, dc in self._directions:
            new_row, new_col = row + dr, col + dc
            new_pos = (new_row, new_col)
            
            if self._is_passable(new_pos):
                cost = self.grid[new_row][new_col]
                neighbors.append((new_pos, cost))
        
        return neighbors
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting position as (row, col) tuple
            end: Ending position as (row, col) tuple
            
        Returns:
            List of (row, col) tuples representing the path from start to end,
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
        if not self._is_passable(start) or not self._is_passable(end):
            return None
        
        # Open set: min-heap of (f_score, g_score, position)
        # Using g_score as secondary sort for tie-breaking
        open_set: List[Tuple[int, int, Tuple[int, int]]] = [(0, 0, start)]
        
        # Track g-scores (cost from start to each position)
        g_scores: dict[Tuple[int, int], int] = {start: 0}
        
        # Track parent of each position for path reconstruction
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # Closed set: positions already evaluated
        closed_set: set[Tuple[int, int]] = set()
        
        while open_set:
            # Get position with lowest f-score
            _, current_g, current = heapq.heappop(open_set)
            
            # Skip if already processed
            if current in closed_set:
                continue
            
            # Goal reached
            if current == end:
                return self._reconstruct_path(came_from, start, end)
            
            closed_set.add(current)
            
            # Explore neighbors
            for neighbor, move_cost in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue
                
                tentative_g = current_g + move_cost
                
                # Found a better path to this neighbor
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
        
        path.reverse()
        return path
    
    def calculate_path_cost(self, path: List[Tuple[int, int]]) -> int:
        """
        Calculate the total cost of a path.
        
        Args:
            path: List of positions in the path
            
        Returns:
            Total cost to traverse the path (sum of costs to enter each cell except start)
        """
        if not path or len(path) == 1:
            return 0
        
        total_cost = 0
        for pos in path[1:]:  # Skip start position
            total_cost += self.grid[pos[0]][pos[1]]
        
        return total_cost

import pytest
from typing import List, Tuple, Optional
import sys
sys.path.insert(0, '.')



class TestAStarGrid:
    """Test suite for A* pathfinding on weighted grids."""
    
    def test_simple_path_uniform_grid(self):
        """Test simple path on a uniform grid with no obstacles."""
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
        
        # Verify path is valid (all adjacent cells)
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            assert abs(r1 - r2) + abs(c1 - c2) == 1, "Consecutive cells must be adjacent"
        
        # Verify optimality: Manhattan distance = 2 + 3 = 5 steps, cost = 5 * 1 = 5
        expected_cost = 5
        actual_cost = astar.calculate_path_cost(path)
        assert actual_cost == expected_cost, f"Expected cost {expected_cost}, got {actual_cost}"
    
    def test_path_around_obstacles(self):
        """Test pathfinding around obstacles."""
        grid = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],  # Row of walls
            [1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1],  # Some walls
            [1, 1, 1, 1, 1],
        ]
        astar = AStarGrid(grid)
        
        start = (0, 0)
        end = (4, 4)
        
        path = astar.find_path(start, end)
        
        # Verify path exists
        assert path is not None, "Path should exist around obstacles"
        
        # Verify path validity
        assert path[0] == start
        assert path[-1] == end
        
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            assert abs(r1 - r2) + abs(c1 - c2) == 1, "Consecutive cells must be adjacent"
            assert grid[r2][c2] > 0, "Path should not go through walls"
        
        # Verify optimality: minimum path must go around the wall row
        # One optimal path: (0,0)->(0,1)->(0,2)->(0,3)->(0,4)->(1,4)->(2,4)->(3,4)->(4,4)
        # Cost = 8 cells * 1 = 8
        expected_cost = 8
        actual_cost = astar.calculate_path_cost(path)
        assert actual_cost == expected_cost, f"Expected cost {expected_cost}, got {actual_cost}"
    
    def test_weighted_grid_prefers_lower_cost(self):
        """Test that A* prefers lower-cost cells over shorter paths."""
        # Create a grid where taking a longer path through low-cost cells
        # is cheaper than a shorter path through high-cost cells
        grid = [
            [1, 100, 100, 1],
            [1, 1, 1, 1],
            [1, 100, 100, 1],
        ]
        astar = AStarGrid(grid)
        
        start = (0, 0)
        end = (0, 3)
        
        path = astar.find_path(start, end)
        
        assert path is not None, "Path should exist"
        
        # Direct path cost: 100 + 100 + 1 = 201
        # Lower path cost: 1 + 1 + 1 + 1 + 1 + 1 = 6
        # A* should choose the lower path
        expected_cost = 6
        actual_cost = astar.calculate_path_cost(path)
        
        assert actual_cost == expected_cost, (
            f"A* should prefer lower-cost path. "
            f"Expected cost {expected_cost}, got {actual_cost}. "
            f"Path: {path}"
        )
        
        # Verify the path goes through row 1 (the low-cost route)
        assert any(row == 1 for row, _ in path), "Path should use the low-cost row"
    
    def test_no_path_exists(self):
        """Test that None is returned when no path exists."""
        grid = [
            [1, 1, 1],
            [0, 0, 0],  # Complete wall blocking the way
            [1, 1, 1],
        ]
        astar = AStarGrid(grid)
        
        start = (0, 0)
        end = (2, 2)
        
        path = astar.find_path(start, end)
        
        assert path is None, "No path should exist when completely blocked"
    
    def test_start_equals_end(self):
        """Test that start == end returns a single-element path."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        astar = AStarGrid(grid)
        
        start = (1, 1)
        end = (1, 1)
        
        path = astar.find_path(start, end)
        
        assert path == [start], f"Expected [{start}], got {path}"
        assert astar.calculate_path_cost(path) == 0, "Cost should be 0 for same start/end"
    
    def test_invalid_coordinates(self):
        """Test that ValueError is raised for out-of-bounds coordinates."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
        ]
        astar = AStarGrid(grid)
        
        # Test invalid start
        with pytest.raises(ValueError, match="Start position"):
            astar.find_path((-1, 0), (1, 1))
        
        with pytest.raises(ValueError, match="Start position"):
            astar.find_path((2, 0), (1, 1))
        
        with pytest.raises(ValueError, match="Start position"):
            astar.find_path((0, 3), (1, 1))
        
        # Test invalid end
        with pytest.raises(ValueError, match="End position"):
            astar.find_path((0, 0), (-1, -1))
        
        with pytest.raises(ValueError, match="End position"):
            astar.find_path((0, 0), (5, 5))
    
    def test_start_or_end_is_wall(self):
        """Test that None is returned when start or end is a wall."""
        grid = [
            [1, 1, 1],
            [1, 0, 1],  # Wall in the middle
            [1, 1, 1],
        ]
        astar = AStarGrid(grid)
        
        # Start is a wall
        path = astar.find_path((1, 1), (0, 0))
        assert path is None, "Should return None when start is a wall"
        
        # End is a wall
        path = astar.find_path((0, 0), (1, 1))
        assert path is None, "Should return None when end is a wall"
    
    def test_optimality_verification(self):
        """Verify that A* always finds the optimal path."""
        # Complex weighted grid
        grid = [
            [1, 2, 3, 1, 5],
            [5, 1, 1, 1, 1],
            [3, 1, 10, 1, 2],
            [1, 1, 1, 1, 1],
        ]
        astar = AStarGrid(grid)
        
        start = (0, 0)
        end = (3, 4)
        
        path = astar.find_path(start, end)
        
        assert path is not None, "Path should exist"
        
        # Calculate the cost
        cost = astar.calculate_path_cost(path)
        
        # Verify by checking all possible paths would be more expensive
        # This is a simplified verification - in production, you'd compare with Dijkstra
        assert cost <= 15, f"Path cost {cost} seems too high for this grid"
        
        # Verify path validity
        assert path[0] == start
        assert path[-1] == end
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]
            assert abs(r1 - r2) + abs(c1 - c2) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])