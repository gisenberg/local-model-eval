import heapq
from typing import List, Tuple, Optional


class AStarGrid:
    """A* pathfinding implementation for weighted 2D grids."""
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize AStarGrid with a 2D grid.
        
        Args:
            grid: 2D list where values represent movement cost.
                  0 = impassable wall, positive int = cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Tuple (row, col) representing starting position
            end: Tuple (row, col) representing ending position
            
        Returns:
            List of (row, col) tuples representing the path from start to end inclusive,
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate coordinates
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start position {start} is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End position {end} is out of bounds")
        
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None
        
        # If start equals end, return single-element path
        if start == end:
            return [start]
        
        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Priority queue: (f_score, g_score, current_node)
        # Using g_score as tie-breaker to ensure deterministic behavior
        open_set = [(0, 0, start)]
        
        # g_score: cost from start to current node
        g_score = {start: 0}
        
        # came_from: for path reconstruction
        came_from = {}
        
        # closed_set: visited nodes
        closed_set = set()
        
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            
            # Skip if already processed
            if current in closed_set:
                continue
            
            # Check if we reached the end
            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            closed_set.add(current)
            
            # Explore neighbors
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                # Check bounds
                if not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols):
                    continue
                
                # Check if wall
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue
                
                # Calculate cost to enter neighbor
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = current_g + move_cost
                
                # If this is a better path to neighbor
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    came_from[neighbor] = current
                    
                    # Calculate f_score = g_score + heuristic
                    heuristic = self._manhattan_distance(neighbor, end)
                    f_score = tentative_g + heuristic
                    
                    # Push to heap
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        # No path found
        return None
    
    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


# Test suite
import pytest


class TestAStarGrid:
    def test_simple_path_uniform_grid(self):
        """Test simple path on uniform grid (all costs = 1)."""
        grid = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (3, 3)
        path = astar.find_path(start, end)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == end
        # Path should be length 7 (3 steps right + 3 steps down + 1 for start)
        assert len(path) == 7
        # Total cost should be 6 (entering 6 cells, each with cost 1)
        total_cost = sum(grid[r][c] for r, c in path[1:])  # Exclude start
        assert total_cost == 6
    
    def test_path_around_obstacles(self):
        """Test path finding around obstacles (walls)."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (3, 3)
        path = astar.find_path(start, end)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == end
        # Verify no walls in path
        for r, c in path:
            assert grid[r][c] != 0
    
    def test_weighted_grid_optimal_path(self):
        """Test that algorithm prefers lower-cost cells."""
        # Create a grid where going through expensive cells is worse
        # than taking a longer but cheaper route
        grid = [
            [1, 100, 100, 1],
            [1, 100, 100, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 3)
        path = astar.find_path(start, end)
        
        assert path is not None
        # Should go down first (cost 1), then right (cost 1), then right (cost 1), then up (cost 1)
        # Path: (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2) -> (2,3)
        # Cost: 1 + 1 + 1 + 1 + 1 = 5
        
        # Calculate actual cost
        total_cost = sum(grid[r][c] for r, c in path[1:])
        assert total_cost == 5
        
        # Verify it doesn't go through expensive cells
        for r, c in path:
            assert grid[r][c] != 100
    
    def test_no_path_exists(self):
        """Test when no path exists due to obstacles."""
        grid = [
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (0, 2)
        path = astar.find_path(start, end)
        
        assert path is None
    
    def test_start_equals_end(self):
        """Test when start and end are the same."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        start = (1, 1)
        end = (1, 1)
        path = astar.find_path(start, end)
        
        assert path == [start]
    
    def test_invalid_coordinates(self):
        """Test that ValueError is raised for out of bounds coordinates."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Test start out of bounds
        with pytest.raises(ValueError):
            astar.find_path((2, 2), (0, 0))
        
        # Test end out of bounds
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (2, 2))
        
        # Test negative coordinates
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (0, 0))