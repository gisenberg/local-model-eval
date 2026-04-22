import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid representation:
        - 0: Wall (impassable)
        - >0: Walkable cell with traversal weight/cost
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid for pathfinding.
        
        Args:
            grid: 2D list of integers where 0 represents walls and 
                  positive integers represent cell weights.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: Tuple of (row, col) representing the starting position.
            end: Tuple of (row, col) representing the target position.
            
        Returns:
            A list of (row, col) tuples representing the optimal path,
            including start and end. Returns None if no valid path exists.
            
        Raises:
            ValueError: If start/end are out of bounds or positioned on a wall (0).
        """
        sr, sc = start
        er, ec = end

        # Input validation
        if not (0 <= sr < self.rows and 0 <= sc < self.cols):
            raise ValueError("Start position out of bounds")
        if not (0 <= er < self.rows and 0 <= ec < self.cols):
            raise ValueError("End position out of bounds")
        if self.grid[sr][sc] == 0:
            raise ValueError("Start position is a wall")
        if self.grid[er][ec] == 0:
            raise ValueError("End position is a wall")

        # Trivial case
        if start == end:
            return [start]

        # A* initialization
        # Priority queue stores: (f_score, tie_breaker, (row, col))
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))
        
        g_score: dict[Tuple[int, int], float] = {start: 0}
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        counter = 1

        # 4-directional movement vectors
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            current_f, _, current = heapq.heappop(open_set)

            # Goal reached
            if current == end:
                path: List[Tuple[int, int]] = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Explore neighbors
            for dr, dc in directions:
                nr, nc = current[0] + dr, current[1] + dc
                
                # Check bounds and walls
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] != 0:
                    tentative_g = g_score[current] + self.grid[nr][nc]
                    
                    if tentative_g < g_score.get((nr, nc), float('inf')):
                        came_from[(nr, nc)] = current
                        g_score[(nr, nc)] = tentative_g
                        
                        # Manhattan heuristic
                        h = abs(nr - er) + abs(nc - ec)
                        f = tentative_g + h
                        
                        heapq.heappush(open_set, (f, counter, (nr, nc)))
                        counter += 1

        return None

import pytest
from typing import List, Tuple, Optional

# Import the class from your module
# 
class AStarGrid:
    # (Paste the class implementation here for standalone testing)
    pass

def test_simple_straight_path() -> None:
    """Test basic pathfinding on an open grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Optimal path length for 3x3 grid is 5 steps (Manhattan distance + 1)
    assert len(path) == 5

def test_path_around_wall() -> None:
    """Test that the algorithm correctly navigates around obstacles."""
    grid = [
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert (1, 1) not in path  # Must avoid the wall

def test_no_valid_path() -> None:
    """Test that None is returned when start and end are disconnected."""
    grid = [
        [1, 0],
        [0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (1, 1))
    assert path is None

def test_start_equals_end() -> None:
    """Test trivial case where start and end are the same."""
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_invalid_positions_raise_value_error() -> None:
    """Test that out-of-bounds and wall positions raise ValueError."""
    grid = [[0, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    # Out of bounds
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (0, 0))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 0))
        
    # On walls
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (0, 1))
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 1), (0, 0))

def test_weighted_optimal_path() -> None:
    """Test that A* chooses lowest cost path, not just shortest steps."""
    # Grid where going through center costs 9, going around costs 1 per cell
    grid = [
        [1, 1, 1],
        [1, 9, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert (1, 1) not in path  # Should avoid high-weight cell
    # Verify total cost is optimal (5 instead of 11)
    total_cost = sum(astar.grid[r][c] for r, c in path[1:])  # Exclude start cell cost
    assert total_cost == 5