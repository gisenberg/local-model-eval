import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """A* pathfinding on a weighted 2D grid."""

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the AStarGrid with a 2D grid.
        
        0 represents a wall (impassable), values >= 1 represent the cost 
        to enter the cell (weighted cells).
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a position is within the grid boundaries."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate the Manhattan distance heuristic between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using the A* algorithm.
        
        Args:
            start: Starting coordinates (row, col)
            end: Ending coordinates (row, col)
            
        Returns:
            A list of coordinates representing the optimal path from start to end,
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds.")

        # If start and end are the same, the path is just the start node
        if start == end:
            return [start]

        # If start or end is on a wall, it's impassable
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Priority queue stores tuples: (f_score, counter, position)
        # The counter is used as a tie-breaker to avoid comparing positions directly
        counter = 0
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, counter, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], int] = {start: 0}
        closed_set: set = set()

        # 4-directional movement: right, down, left, up
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            if current in closed_set:
                continue
            closed_set.add(current)

            current_g = g_score[current]

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)

                if not self._is_valid(neighbor):
                    continue

                # Walls are represented by 0 and are impassable
                if self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue

                neighbor_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = current_g + neighbor_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return None

import pytest

def test_basic_open_path():
    """Test A* finding a basic path in an open grid."""
    grid = [
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ]
    start = (0, 0)
    end = (1, 3)
    path = AStarGrid(grid).find_path(start, end)
    
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    # Optimal path should be exactly 5 nodes long (0,0 -> 0,1 -> 0,2 -> 0,3 -> 1,3)
    assert len(path) == 5

def test_start_equals_end():
    """Test pathfinding when start and end coordinates are identical."""
    grid = [[1, 1], [1, 1]]
    start = (0, 0)
    path = AStarGrid(grid).find_path(start, start)
    
    assert path == [start]

def test_out_of_bounds_raises_value_error():
    """Test that out-of-bounds start or end raises ValueError."""
    grid = [[1, 1], [1, 1]]
    start = (0, 0)
    end = (2, 2)
    
    with pytest.raises(ValueError, match="Start or end position is out of bounds."):
        AStarGrid(grid).find_path(start, end)

def test_no_path_exists():
    """Test that no path is returned when the end is completely walled off."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 1, 1]
    ]
    start = (0, 0)
    end = (2, 2)
    path = AStarGrid(grid).find_path(start, end)
    
    assert path is None

def test_weighted_cells_forces_detour():
    """Test that A* avoids high-cost cells and finds the optimal weighted path."""
    grid = [
        [1, 100, 1],
        [1, 100, 1],
        [1, 1, 1]
    ]
    start = (0, 0)
    end = (2, 2)
    path = AStarGrid(grid).find_path(start, end)
    
    # Optimal path should go down the left edge and across the bottom
    expected_path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    assert path == expected_path

def test_start_on_wall():
    """Test that starting on a wall (0) results in no path."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    start = (0, 0)
    end = (1, 1)
    path = AStarGrid(grid).find_path(start, end)
    
    assert path is None