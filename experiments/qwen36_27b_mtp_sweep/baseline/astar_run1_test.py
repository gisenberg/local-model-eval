import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid cells with value 0 represent impassable walls.
    Positive integer values represent the cost to enter that cell.
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid for pathfinding.
        
        Args:
            grid: 2D list of integers. 0 = wall, >0 = movement cost.
            
        Raises:
            ValueError: If grid is empty or not rectangular.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        if any(len(row) != len(grid[0]) for row in grid):
            raise ValueError("Grid must be rectangular")
            
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic (admissible for 4-directional movement)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using the A* algorithm.
        
        Args:
            start: Tuple (row, col) of the starting position.
            end: Tuple (row, col) of the target position.
            
        Returns:
            A list of (row, col) tuples representing the optimal path, 
            or None if no valid path exists.
            
        Raises:
            ValueError: If start or end is out of bounds or located on a wall.
        """
        # 1. Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds")
            
        # 2. Validate walls at start/end
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall")
            
        # 3. Handle start == end
        if start == end:
            return [start]
            
        # 4. A* Initialization
        # Priority queue stores (f_score, position)
        open_set = [(0, start)]
        g_score = {start: 0}
        came_from = {}
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            
            # Skip stale heap entries
            if current_f > g_score.get(current, float('inf')):
                continue
                
            # Goal reached
            if current == end:
                path = []
                node = current
                while node in came_from:
                    path.append(node)
                    node = came_from[node]
                path.append(start)
                return path[::-1]
                
            r, c = current
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                # Check bounds
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                    
                # Check walls
                if self.grid[nr][nc] == 0:
                    continue
                    
                move_cost = self.grid[nr][nc]
                tentative_g = g_score[current] + move_cost
                
                # Found a better path to neighbor
                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    f_score = tentative_g + self._heuristic((nr, nc), end)
                    heapq.heappush(open_set, (f_score, (nr, nc)))
                    
        # No path found
        return None

import pytest
from typing import List, Tuple, Optional

def test_basic_uniform_grid_path() -> None:
    """Test standard pathfinding on a uniform cost grid."""
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
    assert len(path) == 5  # Manhattan distance 4 => 5 nodes

def test_start_equals_end() -> None:
    """Test handling when start and end positions are identical."""
    grid = [[1, 2], [3, 4]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_out_of_bounds_raises_value_error() -> None:
    """Test that out-of-bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_blocks_path_returns_none() -> None:
    """Test that a completely blocked path returns None."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_weighted_grid_optimal_path() -> None:
    """Test that A* correctly chooses the lower-cost path over a shorter geometric path."""
    # High cost cells in the middle force path around the bottom
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    # Optimal path avoids the 10-cost cells
    assert (0, 1) not in path
    assert (1, 1) not in path
    # Verify total cost is 4 (entering 4 cells of cost 1)
    total_cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert total_cost == 4

def test_wall_at_start_or_end_raises_value_error() -> None:
    """Test that placing start or end on a wall raises ValueError."""
    grid = [[1, 0], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError, match="Start position is a wall"):
        astar.find_path((0, 1), (1, 1))
    with pytest.raises(ValueError, match="End position is a wall"):
        astar.find_path((0, 0), (0, 1))