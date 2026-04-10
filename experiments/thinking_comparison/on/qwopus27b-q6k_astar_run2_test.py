from typing import List, Tuple, Optional, Dict, Set
import heapq


class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.
    
    Grid values represent movement cost:
    - 0: impassable wall
    - positive int: cost to enter that cell
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize AStarGrid with a 2D grid.
        
        Args:
            grid: 2D list where 0 represents a wall and positive integers 
                  represent the cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid bounds."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols
    
    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a wall (cost 0)."""
        r, c = pos
        return self.grid[r][c] == 0
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting position (row, col)
            end: Ending position (row, col)
            
        Returns:
            List of positions from start to end inclusive, or None if no path exists.
            
        Raises:
            ValueError: If start or end is out of bounds.
        """
        # Validate inputs
        if not self._is_valid(start):
            raise ValueError(f"Start position {start} is out of bounds")
        if not self._is_valid(end):
            raise ValueError(f"End position {end} is out of bounds")
        
        # Check if start or end is a wall
        if self._is_wall(start) or self._is_wall(end):
            return None
        
        # Edge case: start equals end
        if start == end:
            return [start]
        
        # A* initialization
        open_set = [(self._heuristic(start, end), start)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        closed_set: Set[Tuple[int, int]] = set()
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            # If we reached the end, reconstruct path
            if current == end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            # Skip if already processed with better cost
            if current in closed_set:
                continue
            closed_set.add(current)
            
            # Explore neighbors
            for dr, dc in self.directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not self._is_valid(neighbor) or self._is_wall(neighbor) or neighbor in closed_set:
                    continue
                
                # Cost to enter neighbor is the value of the neighbor cell
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    came_from[neighbor] = current
                    heapq.heappush(open_set, (f_score, neighbor))
        
        return None


# Tests
import pytest


class TestAStarGrid:
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
        # Should be 7 steps (start + 6 moves)
        assert len(path) == 7
        # Check validity: each step should be adjacent
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            assert abs(r1 - r2) + abs(c1 - c2) == 1
    
    def test_path_around_obstacles(self):
        """Test path finding around obstacles (walls)."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (3, 3))
        
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (3, 3)
        # Verify path doesn't hit walls
        for pos in path:
            assert grid[pos[0]][pos[1]] != 0
    
    def test_weighted_grid_optimal_path(self):
        """Test that path prefers lower-cost cells."""
        # Create a grid where going through expensive cells is worse
        grid = [
            [1, 10, 10, 1],
            [1, 10, 10, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 3))
        
        assert path is not None
        # Calculate total cost (sum of costs to enter each cell, excluding start)
        total_cost = sum(grid[r][c] for r, c in path[1:])
        # Optimal path should go down first column then across bottom row: cost 5
        # Alternative path through top costs 23
        assert total_cost == 5
    
    def test_no_path_exists(self):
        """Test when no path exists due to walls."""
        grid = [
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        assert path is None
    
    def test_start_equals_end(self):
        """Test when start equals end."""
        grid = [[1, 1], [1, 1]]
        astar = AStarGrid(grid)
        path = astar.find_path((1, 1), (1, 1))
        assert path == [(1, 1)]
    
    def test_invalid_coordinates(self):
        """Test that invalid coordinates raise ValueError."""
        grid = [[1, 1], [1, 1]]
        astar = AStarGrid(grid)
        
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (1, 1))
        
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (2, 2))
        
        with pytest.raises(ValueError):
            astar.find_path((0, -1), (1, 1))