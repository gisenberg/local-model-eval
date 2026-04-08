import heapq
from typing import List, Tuple, Optional


class AStarGrid:
    """A* pathfinding algorithm on a weighted 2D grid."""
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D grid.
        
        Args:
            grid: 2D list where 0 represents a wall and positive integers represent
                  the cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
    
    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting coordinates (row, col)
            end: Ending coordinates (row, col)
            
        Returns:
            List of coordinates from start to end inclusive, or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate coordinates
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start coordinates {start} out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End coordinates {end} out of bounds")
        
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
        
        # Start equals end
        if start == end:
            return [start]
        
        # A* algorithm
        # Priority queue stores (f_score, g_score, current_node)
        open_set = []
        heapq.heappush(open_set, (0, 0, start))
        
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: dict[Tuple[int, int], int] = {start: 0}
        f_score: dict[Tuple[int, int], int] = {start: self._manhattan(start, end)}
        
        while open_set:
            _, current_g, current = heapq.heappop(open_set)
            
            if current == end:
                return self._reconstruct_path(came_from, end)
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue
                
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._manhattan(neighbor, end)
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, tentative_g, neighbor))
        
        return None
    
    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid 4-directional neighbors of a position."""
        row, col = pos
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                neighbors.append((nr, nc))
        return neighbors
    
    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from end to start using came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


# Tests
import pytest


def test_simple_path_uniform_grid():
    """Test simple path on a uniform cost grid."""
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
    
    # Verify path validity (adjacent cells)
    for i in range(len(path) - 1):
        assert abs(path[i][0] - path[i+1][0]) + abs(path[i][1] - path[i+1][1]) == 1
    
    # Verify optimality (Manhattan distance is 4, each step costs 1)
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4


def test_path_around_obstacles():
    """Test pathfinding around obstacles."""
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 3))
    
    assert path is not None
    # Path should not go through walls
    assert (1, 1) not in path
    assert (1, 2) not in path
    
    # Verify path validity
    for i in range(len(path) - 1):
        assert abs(path[i][0] - path[i+1][0]) + abs(path[i][1] - path[i+1][1]) == 1


def test_weighted_grid():
    """Test that algorithm prefers lower-cost cells over shorter paths."""
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    
    # Should avoid expensive middle column (cost 10)
    # Optimal path: (0,0)->(1,0)->(2,0)->(2,1)->(2,2) with cost 4
    # Alternative: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) with cost 13
    assert (0, 1) not in path
    assert (1, 1) not in path
    
    # Verify optimality
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4


def test_no_path_exists():
    """Test when no path exists due to obstacles or walls at start/end."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    
    # Fully blocked path
    path = astar.find_path((0, 0), (0, 2))
    assert path is None
    
    # Wall at start
    path = astar.find_path((1, 1), (2, 2))  # (1,1) is a wall
    assert path is None
    
    # Wall at end
    path = astar.find_path((0, 0), (1, 1))  # (1,1) is a wall
    assert path is None


def test_start_equals_end():
    """Test when start and end are the same."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]


def test_invalid_coordinates():
    """Test that invalid coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))
    with pytest.raises(ValueError):
        astar.find_path((0, -1), (0, 0))
    with pytest.raises(ValueError):
        astar.find_path((0, 2), (0, 0))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])