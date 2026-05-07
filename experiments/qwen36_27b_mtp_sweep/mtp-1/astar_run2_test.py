import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Movement is 4-directional (up, down, left, right). 
    Cell values represent the cost to ENTER that cell. 
    0 represents an impassable wall. Positive integers represent weighted terrain.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the pathfinding grid.
        
        :param grid: 2D list of integers. 0 = wall, >0 = movement cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* search.
        
        :param start: (row, col) tuple for the starting position.
        :param end: (row, col) tuple for the target position.
        :return: List of coordinates from start to end, or None if no path exists.
        :raises ValueError: If start or end coordinates are out of grid bounds.
        """
        # 1. Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds")

        # 2. Handle walls at start/end
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # 3. Handle start == end
        if start == end:
            return [start]

        # A* initialization
        # Priority queue stores: (f_score, tie_breaker_counter, (row, col))
        counter = 0
        open_set: List[Tuple[float, int, Tuple[int, int]]] = [(0, counter, start)]
        
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        
        # 4-directional movement
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open_set:
            _, _, current = heapq.heappop(open_set)

            # Goal reached
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            cr, cc = current
            for dr, dc in directions:
                nr, nc = cr + dr, cc + dc

                # Skip out-of-bounds
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                # Skip walls
                if self.grid[nr][nc] == 0:
                    continue

                # Cost to move into the neighbor cell
                move_cost = self.grid[nr][nc]
                tentative_g = g_score[current] + move_cost

                # Update if we found a cheaper path to this neighbor
                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    
                    # f_score = g_score + Manhattan heuristic
                    f_score = tentative_g + abs(nr - end[0]) + abs(nc - end[1])
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, (nr, nc)))

        # Queue exhausted without reaching end
        return None

import pytest

def test_basic_path():
    """Test standard pathfinding on an open grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5  # Optimal Manhattan distance + 1
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[5]]
    assert AStarGrid(grid).find_path((0, 0), (0, 0)) == [(0, 0)]

def test_out_of_bounds_start():
    """Test ValueError raised for invalid start coordinates."""
    grid = [[1, 1], [1, 1]]
    with pytest.raises(ValueError, match="Start position is out of bounds"):
        AStarGrid(grid).find_path((-1, 0), (1, 1))

def test_out_of_bounds_end():
    """Test ValueError raised for invalid end coordinates."""
    grid = [[1, 1], [1, 1]]
    with pytest.raises(ValueError, match="End position is out of bounds"):
        AStarGrid(grid).find_path((0, 0), (2, 2))

def test_walls_block_path():
    """Test that walls correctly block paths and return None."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    assert AStarGrid(grid).find_path((0, 0), (0, 2)) is None

def test_weighted_optimal_path():
    """Test that A* chooses the lowest-cost path over the shortest geometric path."""
    # Direct path through middle costs 10+1+1+1 = 13
    # Path around costs 1+1+1+1 = 4
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    path = AStarGrid(grid).find_path((0, 0), (2, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]