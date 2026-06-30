import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid cells with value 0 represent walls (impassable).
    Positive integer values represent traversal costs (weights).
    Movement is restricted to 4 directions (up, down, left, right).
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid for pathfinding.
        
        :param grid: 2D list of integers where 0 is a wall and >0 is a traversal cost.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid must be non-empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if coordinates are within grid boundaries."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if a cell is a wall (value 0)."""
        return self.grid[pos[0]][pos[1]] == 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic (admissible for 4-directional movement)."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using the A* algorithm.
        
        :param start: (row, col) tuple of starting position.
        :param end: (row, col) tuple of target position.
        :return: List of (row, col) tuples representing the optimal path, or None if unreachable.
        :raises ValueError: If start/end are out of bounds or positioned on walls.
        """
        # 1. Validate bounds
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds.")
        
        # 2. Validate walls
        if self._is_wall(start) or self._is_wall(end):
            raise ValueError("Start or end position is a wall.")
            
        # 3. Handle start == end
        if start == end:
            return [start]

        # A* Initialization
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        counter = 0  # Tie-breaker for heapq when f_scores are equal
        heapq.heappush(open_set, (self._heuristic(start, end), counter, start))
        
        g_score: dict[Tuple[int, int], int] = {start: 0}
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

        while open_set:
            _, _, current = heapq.heappop(open_set)

            # Goal reached
            if current == end:
                path: List[Tuple[int, int]] = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Reverse to get start -> end order

            # Explore neighbors
            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not self._is_valid(neighbor) or self._is_wall(neighbor):
                    continue

                # Cost to enter the neighbor cell
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost

                # Found a better path to neighbor
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        # No path found
        return None

import pytest

def test_basic_pathfinding() -> None:
    """Test standard pathfinding on an unweighted grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

def test_start_equals_end() -> None:
    """Test immediate return when start and end are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_exists() -> None:
    """Test return None when destination is completely blocked."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_out_of_bounds_raises_value_error() -> None:
    """Test ValueError for coordinates outside grid dimensions."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (2, 2))

def test_wall_positions_raise_value_error() -> None:
    """Test ValueError when start or end is placed on a wall."""
    grid = [
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 1), (2, 2))
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (0, 1))

def test_weighted_optimal_path() -> None:
    """Test that A* correctly chooses the lowest-cost path over a shorter geometric path."""
    # Direct path cost: 100 + 1 = 101
    # Detour path cost: 1 + 1 + 1 + 1 = 4
    grid = [
        [1, 100, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is not None
    assert (0, 1) not in path  # Should avoid the expensive cell
    
    # Verify optimality by summing traversal costs (excluding start cell)
    total_cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert total_cost == 4