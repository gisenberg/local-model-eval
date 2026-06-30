import heapq
from typing import List, Tuple, Optional, Dict, Set

class AStarGrid:
    """
    A* pathfinding algorithm on a 2D weighted grid.
    
    Grid conventions:
    - Values > 0: Walkable cells. The value represents the movement cost to ENTER the cell.
    - Value == 0: Wall (impassable).
    - Movement: 4-directional (up, down, left, right).
    - Heuristic: Manhattan distance (admissible for weights >= 1).
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid.
        
        Args:
            grid: 2D list of integers representing cell weights. 0 denotes a wall.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_walkable(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within bounds and not a wall."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] > 0

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two positions."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: Starting coordinates (row, col).
            end: Target coordinates (row, col).
            
        Returns:
            List of coordinates from start to end, or None if no path exists.
            
        Raises:
            ValueError: If start or end is out of bounds or on a wall.
        """
        if not self._is_walkable(start) or not self._is_walkable(end):
            raise ValueError("Start or end position is out of bounds or on a wall.")
        if start == end:
            return [start]

        # Priority queue stores tuples: (f_score, tie_breaker, position)
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (self._manhattan(start, end), 0, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        closed_set: Set[Tuple[int, int]] = set()
        counter = 1  # Tie-breaker to avoid comparing tuples in heapq

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Explore 4-directional neighbors
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = current[0] + dr, current[1] + dc
                neighbor = (nr, nc)

                if not self._is_walkable(neighbor):
                    continue

                # Cost to enter the neighbor cell
                tentative_g = g_score[current] + self.grid[nr][nc]
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._manhattan(neighbor, end)
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1

        return None

import pytest

def test_basic_open_grid_path():
    """Test finding a valid path in an open grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Verify 4-directional adjacency
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1

def test_wall_detour():
    """Test that path correctly routes around walls."""
    grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert (1, 1) not in path  # Wall should be avoided

def test_start_equals_end():
    """Test immediate return when start and end are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_returns_none():
    """Test that isolated start/end returns None."""
    grid = [[1, 0, 1], [0, 0, 0], [1, 0, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_invalid_positions_raise_value_error():
    """Test that out-of-bounds and wall positions raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))  # Out of bounds
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0)) # Out of bounds
        
    grid_wall = [[1, 0], [1, 1]]
    astar_wall = AStarGrid(grid_wall)
    with pytest.raises(ValueError):
        astar_wall.find_path((0, 1), (1, 1)) # Start on wall
    with pytest.raises(ValueError):
        astar_wall.find_path((0, 0), (0, 1)) # End on wall

def test_weighted_optimality():
    """Test that A* chooses minimum cost path over shortest step count."""
    # Direct path through center costs 100, detour costs 4
    grid = [
        [1, 1, 1],
        [1, 100, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert (1, 1) not in path  # Should avoid heavy cell
    
    # Calculate total movement cost (excluding start cell)
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4  # Optimal detour cost