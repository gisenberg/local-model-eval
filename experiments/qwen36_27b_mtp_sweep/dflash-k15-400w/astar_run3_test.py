import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm for a weighted 2D grid.
    
    The grid is represented as a list of lists of integers where:
    - 0 represents a wall (impassable)
    - Positive integers represent the traversal cost to enter that cell
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the pathfinding grid.
        
        Args:
            grid: 2D list of integers representing cell costs. 0 = wall.
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
            start: Tuple (row, col) of the starting position.
            end: Tuple (row, col) of the target position.
            
        Returns:
            A list of (row, col) tuples representing the optimal path,
            or None if no valid path exists.
            
        Raises:
            ValueError: If start/end are out of bounds or positioned on a wall.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds")
            
        # Validate walls
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            raise ValueError("Start or end position is a wall")
            
        # Handle immediate success
        if start == end:
            return [start]

        # Priority queue: (f_score, tie_breaker, (row, col))
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))
        
        g_score: dict[Tuple[int, int], int] = {start: 0}
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        closed_set: set[Tuple[int, int]] = set()
        
        # 4-directional movement: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        tie_breaker = 0

        while open_set:
            f, _, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            if current in closed_set:
                continue
            closed_set.add(current)

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                nr, nc = neighbor

                # Skip out-of-bounds
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                # Skip walls
                if self.grid[nr][nc] == 0:
                    continue
                # Skip already processed nodes
                if neighbor in closed_set:
                    continue

                move_cost = self.grid[nr][nc]
                tentative_g = g_score[current] + move_cost

                # Update if we found a cheaper path to neighbor
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tie_breaker, neighbor))
                    tie_breaker += 1

        return None

    @staticmethod
    def _heuristic(pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """Manhattan distance heuristic for 4-directional movement."""
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    @staticmethod
    def _reconstruct_path(came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Backtrack from end to start using the came_from map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

import pytest

def test_basic_pathfinding():
    """Test standard pathfinding on an open grid."""
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
    # Optimal path length for 3x3 grid is 5 steps (Manhattan distance + 1)
    assert len(path) == 5

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_no_path_exists():
    """Test grid where destination is completely blocked by walls."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
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
    """Test that starting or ending on a wall raises ValueError."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((1, 0), (0, 1))

def test_weighted_optimal_path():
    """Test that A* chooses lower-cost path over shorter Manhattan distance."""
    # Direct top path cost: 5 + 5 + 5 = 15
    # Bottom detour cost: 1 + 1 + 1 + 5 = 9
    grid = [
        [1, 5, 5, 5],
        [1, 1, 1, 1],
        [1, 5, 5, 5]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 3))
    
    assert path is not None
    # Verify it took the bottom route
    assert (1, 0) in path and (1, 3) in path
    
    # Verify total cost matches optimal weighted path
    total_cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert total_cost == 9