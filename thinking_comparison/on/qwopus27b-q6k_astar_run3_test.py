import heapq
from typing import List, Tuple, Optional


class AStarGrid:
    """A* pathfinding implementation on a weighted 2D grid."""
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize AStarGrid with a 2D grid.
        
        Args:
            grid: 2D list where 0 represents impassable wall, 
                  positive integers represent movement cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting position as (row, col) tuple
            end: Ending position as (row, col) tuple
            
        Returns:
            List of (row, col) coordinates representing the path from start to end,
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate coordinates are within bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start position {start} is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End position {end} is out of bounds")
        
        # Check if start or end are walls (cost 0)
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None
        
        # If start equals end, return just the start position
        if start == end:
            return [start]
        
        # Directions: up, down, left, right (4-directional)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        # Priority queue: (f_score, g_score, node)
        # f_score = g_score + heuristic
        open_set: List[Tuple[int, int, Tuple[int, int]]] = [(0, 0, start)]
        
        # Track cost from start to each node
        g_score: dict[Tuple[int, int], int] = {start: 0}
        
        # Track visited nodes
        closed_set: set[Tuple[int, int]] = set()
        
        # Track parent for path reconstruction
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        
        while open_set:
            f, g, current = heapq.heappop(open_set)
            
            # Skip if already processed
            if current in closed_set:
                continue
                
            # Goal reached - reconstruct path
            if current == end:
                path: List[Tuple[int, int]] = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            closed_set.add(current)
            
            # Explore neighbors
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                # Check bounds
                if not (0 <= neighbor[0] < self.rows and 0 <= neighbor[1] < self.cols):
                    continue
                
                # Skip walls
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue
                
                # Skip already processed nodes
                if neighbor in closed_set:
                    continue
                
                # Cost to enter neighbor cell
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g + move_cost
                
                # Update if better path found
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._manhattan_distance(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    came_from[neighbor] = current
        
        return None
    
    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


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
        
        # Verify path is contiguous (each step adjacent to previous)
        for i in range(1, len(path)):
            r1, c1 = path[i-1]
            r2, c2 = path[i]
            assert abs(r1 - r2) + abs(c1 - c2) == 1
        
        # Verify optimality (Manhattan distance is 4, so 5 nodes, cost 4)
        assert len(path) == 5
        total_cost = sum(grid[r][c] for r, c in path[1:])  # Skip start
        assert total_cost == 4
    
    def test_path_around_obstacles(self):
        """Test path finding around obstacles (walls)."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        
        # Verify no walls in path
        for r, c in path:
            assert grid[r][c] != 0
        
        # Verify optimality (cost should be 4)
        total_cost = sum(grid[r][c] for r, c in path[1:])
        assert total_cost == 4
    
    def test_weighted_grid(self):
        """Test that path prefers lower-cost cells."""
        grid = [
            [1, 10, 1, 1],
            [1, 10, 10, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        
        # Verify path doesn't use high-cost cells (10)
        for r, c in path:
            assert grid[r][c] != 10
        
        # Verify optimality (cost should be 4 via bottom-left route)
        total_cost = sum(grid[r][c] for r, c in path[1:])
        assert total_cost == 4
    
    def test_no_path_exists(self):
        """Test when no path exists (fully blocked)."""
        grid = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is None
    
    def test_start_equals_end(self):
        """Test when start equals end."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((1, 1), (1, 1))
        
        assert path == [(1, 1)]
    
    def test_invalid_coordinates(self):
        """Test invalid coordinates raise ValueError."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Test start out of bounds
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (2, 2))
        with pytest.raises(ValueError):
            astar.find_path((0, -1), (2, 2))
        with pytest.raises(ValueError):
            astar.find_path((3, 0), (2, 2))
        with pytest.raises(ValueError):
            astar.find_path((0, 3), (2, 2))
        
        # Test end out of bounds
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (-1, 0))
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (3, 0))