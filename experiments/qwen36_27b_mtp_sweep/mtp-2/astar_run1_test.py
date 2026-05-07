import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Walls are represented by 0. Traversable cells contain positive numeric weights
    representing the cost to enter that cell. Movement is restricted to 4 directions.
    """
    
    def __init__(self, grid: List[List[float]]) -> None:
        """
        Initialize the pathfinding grid.
        
        Args:
            grid: 2D list where 0 represents a wall and positive numbers represent 
                  traversal costs. All rows must have the same length.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        if any(len(row) != len(grid[0]) for row in grid):
            raise ValueError("All rows in the grid must have the same length.")
            
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using the A* algorithm.
        
        Args:
            start: (row, col) tuple for the starting position.
            end: (row, col) tuple for the target position.
            
        Returns:
            A list of (row, col) tuples representing the optimal path from start to end,
            or None if no valid path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds or land on a wall.
        """
        # 1. Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds.")
            
        # 2. Validate walls
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall.")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall.")
            
        # 3. Handle start == end
        if start == end:
            return [start]
            
        # 4. A* Initialization
        # Priority queue stores (f_score, counter, (row, col))
        # Counter breaks ties to avoid comparing coordinate tuples
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, 0, start))
        
        g_score: dict[Tuple[int, int], float] = {start: 0.0}
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        counter = 1
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while open_set:
            f, _, current = heapq.heappop(open_set)
            
            # Goal reached
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
                
            # Skip stale heap entries
            if f > g_score[current]:
                continue
                
            # Explore neighbors
            for dr, dc in directions:
                nr, nc = current[0] + dr, current[1] + dc
                
                # Check bounds and walls
                if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid[nr][nc] > 0:
                    move_cost = self.grid[nr][nc]
                    tentative_g = g_score[current] + move_cost
                    
                    if tentative_g < g_score.get((nr, nc), float('inf')):
                        came_from[(nr, nc)] = current
                        g_score[(nr, nc)] = tentative_g
                        
                        # Manhattan heuristic (admissible & consistent for 4-dir grids)
                        h = abs(nr - end[0]) + abs(nc - end[1])
                        f_new = tentative_g + h
                        
                        heapq.heappush(open_set, (f_new, counter, (nr, nc)))
                        counter += 1
                        
        return None

import pytest

def test_basic_path():
    """Test straightforward pathfinding on an open grid."""
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
    # Manhattan distance is 4 steps -> 5 nodes in path
    assert len(path) == 5

def test_start_equals_end():
    """Test that start == end returns a single-element path."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_exists():
    """Test that surrounded targets return None."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_out_of_bounds_start():
    """Test ValueError for out-of-bounds start coordinates."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))

def test_wall_at_end():
    """Test ValueError when target is a wall."""
    grid = [
        [1, 1],
        [1, 0]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))

def test_weighted_optimal_path():
    """Test that A* chooses a longer but cheaper path over a direct expensive one."""
    grid = [
        [1, 100, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is not None
    # Direct path cost: 100 + 1 = 101
    # Optimal detour cost: 1 + 1 + 1 + 1 = 4
    assert (0, 1) not in path, "Path should avoid the high-cost cell"
    
    # Verify actual path cost matches optimal
    path_cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert path_cost == 4