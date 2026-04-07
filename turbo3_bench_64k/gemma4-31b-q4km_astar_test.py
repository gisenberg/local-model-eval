import heapq
from typing import List, Tuple, Optional


class AStarGrid:
    """A* pathfinding implementation on a weighted 2D grid."""
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize A* pathfinding on a weighted 2D grid.
        
        Args:
            grid: 2D list where values represent movement cost.
                  0 = impassable wall, positive int = cost to enter cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within bounds and not a wall."""
        r, c = pos
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return self.grid[r][c] > 0
        return False
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting position (row, col)
            end: Ending position (row, col)
            
        Returns:
            List of coordinates from start to end inclusive, or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Check bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start position {start} is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End position {end} is out of bounds")
        
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        
        # If start equals end
        if start == end:
            return [start]
        
        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Priority queue: (f_score, g_score, current_pos)
        # Using g_score as tiebreaker to ensure consistent ordering
        open_set = [(self._heuristic(start, end), 0, start)]
        
        # Track best g_score for each node
        g_scores = {start: 0}
        
        # Track where we came from for path reconstruction
        came_from = {}
        
        # Closed set (visited)
        closed_set = set()
        
        while open_set:
            f_score, g_score, current = heapq.heappop(open_set)
            
            # If we've already processed this node with a better score, skip
            if current in closed_set:
                continue
            
            # If this g_score is worse than what we already found, skip
            if g_score > g_scores.get(current, float('inf')):
                continue
            
            # Found the goal
            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]  # Reverse to get start to end
            
            closed_set.add(current)
            
            # Explore neighbors
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if neighbor in closed_set:
                    continue
                
                # Check if valid (in bounds and not wall)
                if not self._is_valid(neighbor):
                    continue
                
                # Cost to move to neighbor (value in grid)
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score + move_cost
                
                # If this path to neighbor is better than any previous one
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    came_from[neighbor] = current
        
        # No path found
        return None


# Test suite
import pytest


class TestAStarGrid:
    def test_simple_path_uniform_grid(self):
        """Test simple path on uniform grid with cost 1."""
        grid = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        # Manhattan distance is 4, so path length should be 5 (including start)
        assert len(path) == 5
        # Verify optimality: total cost should be 4 (4 moves of cost 1)
        total_cost = sum(grid[r][c] for r, c in path[1:])  # Exclude start
        assert total_cost == 4
    
    def test_path_around_obstacles(self):
        """Test path finding around obstacles (walls)."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 3))
        
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 3)
        # Verify no walls in path
        for r, c in path:
            assert grid[r][c] != 0
        # Verify path is valid (consecutive cells are adjacent)
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            assert abs(r1 - r2) + abs(c1 - c2) == 1
    
    def test_weighted_grid(self):
        """Test that path prefers lower-cost cells."""
        grid = [
            [1, 1, 1, 1],
            [1, 100, 100, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 3))
        
        assert path is not None
        # Should avoid the expensive middle row
        for r, c in path:
            if r == 1:
                assert c != 1 and c != 2
        
        # Calculate cost - optimal path goes around edges with cost 1 each
        total_cost = sum(grid[r][c] for r, c in path[1:])
        assert total_cost == 5
    
    def test_no_path_exists(self):
        """Test when no path exists due to walls."""
        grid = [
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (0, 2))
        assert path is None
        
        # Also test start/end being walls
        grid2 = [
            [0, 1],
            [1, 1]
        ]
        astar2 = AStarGrid(grid2)
        assert astar2.find_path((0, 0), (1, 1)) is None  # Start is wall
        
        grid3 = [
            [1, 1],
            [1, 0]
        ]
        astar3 = AStarGrid(grid3)
        assert astar3.find_path((0, 0), (1, 1)) is None  # End is wall
    
    def test_start_equals_end(self):
        """Test when start equals end."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((1, 1), (1, 1))
        assert path == [(1, 1)]
    
    def test_invalid_coordinates(self):
        """Test that invalid coordinates raise ValueError."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Test out of bounds start
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (1, 1))
        
        # Test out of bounds end
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (2, 2))
        
        # Test negative coordinates
        with pytest.raises(ValueError):
            astar.find_path((0, -1), (1, 1))