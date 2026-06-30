import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid values represent the cost to enter a cell. A value of 0 indicates a wall (impassable).
    Movement is restricted to 4 directions (up, down, left, right).
    Uses Manhattan distance as the admissible and consistent heuristic.
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.
        
        Args:
            grid: 2D list of integers where 0 represents a wall and positive integers
                  represent the traversal cost of the cell.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        if any(len(row) != len(grid[0]) for row in grid):
            raise ValueError("Grid must be rectangular.")
            
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* search.
        
        Args:
            start: (row, col) tuple for the starting position.
            end: (row, col) tuple for the target position.
            
        Returns:
            A list of (row, col) tuples representing the optimal path from start to end,
            or None if no valid path exists.
            
        Raises:
            ValueError: If start or end is out of bounds or located on a wall.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds.")
            
        # Validate walls
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall.")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall.")
            
        # Handle start == end
        if start == end:
            return [start]

        # Priority queue: (f_score, counter, (row, col))
        # counter breaks ties deterministically and avoids tuple comparison errors
        pq = [(self._manhattan(start, end), 0, start)]
        counter = 1
        
        g_score = {start: 0}
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        
        while pq:
            f, _, current = heapq.heappop(pq)
            
            if current == end:
                return self._reconstruct_path(came_from, current)
                
            # Explore 4-directional neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = current[0] + dr, current[1] + dc
                
                # Skip out-of-bounds and walls
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                if self.grid[nr][nc] == 0:
                    continue
                    
                move_cost = self.grid[nr][nc]
                tentative_g = g_score[current] + move_cost
                
                # Found a better path to neighbor
                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + self._manhattan((nr, nc), end)
                    heapq.heappush(pq, (f_score, counter, (nr, nc)))
                    counter += 1
                    
        return None

    @staticmethod
    def _manhattan(a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two grid coordinates."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _reconstruct_path(came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Backtrack from end to start using the came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]


# ========================
# Pytest Test Suite
# ========================
import pytest

def test_basic_pathfinding():
    """Test standard pathfinding on a uniform grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Manhattan distance is 4 steps -> 5 nodes
    assert len(path) == 5

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[5]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_no_path_exists():
    """Test when target is completely surrounded by walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_at_start_or_end_raises_value_error():
    """Test that placing start/end on a wall raises ValueError."""
    grid_start_wall = [[0, 1], [1, 1]]
    grid_end_wall = [[1, 0], [1, 1]]
    
    with pytest.raises(ValueError, match="wall"):
        AStarGrid(grid_start_wall).find_path((0, 0), (1, 1))
    with pytest.raises(ValueError, match="wall"):
        AStarGrid(grid_end_wall).find_path((0, 0), (0, 1))

def test_weighted_optimality():
    """Test that A* chooses the lower-cost path over the shorter path."""
    # Direct path through center costs 9, path around costs 1+1+1+1+1 = 5
    grid = [
        [1, 1, 1],
        [1, 9, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    # Optimal path must avoid the expensive center cell
    assert (1, 1) not in path
    # Verify total cost matches the cheaper route
    total_cost = sum(grid[r][c] for r, c in path[1:])  # Exclude start cell cost
    assert total_cost == 5