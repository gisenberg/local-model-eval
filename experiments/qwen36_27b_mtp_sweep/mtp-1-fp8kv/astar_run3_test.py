import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid values represent the cost to ENTER a cell. 
    A value of 0 denotes a wall (impassable). Positive integers denote traversal weights.
    """
    def __init__(self, grid: List[List[int]]) -> None:
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using the A* algorithm.
        
        Args:
            start: Tuple (row, col) representing the starting position.
            end: Tuple (row, col) representing the target position.
            
        Returns:
            A list of (row, col) tuples representing the optimal path, 
            or None if no valid path exists.
            
        Raises:
            ValueError: If start or end is out of bounds or positioned on a wall.
        """
        # 1. Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds.")

        # 2. Validate walls at start/end
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall.")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall.")

        # 3. Handle start == end
        if start == end:
            return [start]

        # A* initialization
        # Priority queue stores: (f_score, tie_breaker_counter, (row, col))
        open_set: List[Tuple[float, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0.0, 0, start))
        
        g_score: dict[Tuple[int, int], float] = {start: 0.0}
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        counter = 1
        
        # 4-directional movement
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_set:
            current_f, _, current = heapq.heappop(open_set)

            # Goal reached: reconstruct path
            if current == end:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Lazy deletion: skip if we've already found a better path to this node
            if current_f > g_score.get(current, float('inf')):
                continue

            for dr, dc in directions:
                nr, nc = current[0] + dr, current[1] + dc

                # Bounds check
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue

                # Wall check
                if self.grid[nr][nc] == 0:
                    continue

                # Cost to enter neighbor
                move_cost = self.grid[nr][nc]
                tentative_g = g_score[current] + move_cost

                # Update if better path found
                if tentative_g < g_score.get((nr, nc), float('inf')):
                    came_from[(nr, nc)] = current
                    g_score[(nr, nc)] = tentative_g
                    # Manhattan heuristic
                    h_score = abs(nr - end[0]) + abs(nc - end[1])
                    f_score = tentative_g + h_score
                    heapq.heappush(open_set, (f_score, counter, (nr, nc)))
                    counter += 1

        # No path found
        return None

import pytest

def test_basic_pathfinding():
    """Test standard pathfinding on an unweighted grid."""
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
    # Manhattan distance is 4 steps, so path length should be 5 nodes
    assert len(path) == 5

def test_start_equals_end():
    """Test when start and end positions are identical."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_no_path_exists():
    """Test that None is returned when the target is completely blocked by walls."""
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
    """Test that placing start/end on a wall raises ValueError."""
    grid = [
        [0, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((0, 0), (1, 1))
    with pytest.raises(ValueError, match="wall"):
        astar.find_path((1, 1), (0, 0))

def test_weighted_optimality():
    """Test that A* correctly avoids high-weight cells when a cheaper alternative exists."""
    # Grid layout:
    # 1 1 1
    # 1 9 1
    # 1 1 1
    # Direct path through center costs 9, detour costs 4
    grid = [
        [1, 1, 1],
        [1, 9, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    # Verify the expensive cell (1,1) is avoided
    assert (1, 1) not in path
    
    # Calculate total traversal cost (sum of weights of all cells except start)
    path_cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert path_cost == 4  # Optimal cost is 4, not 10