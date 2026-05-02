import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    """
    Implements the A* search algorithm for finding the optimal path 
    on a weighted 2D grid with obstacles.
    """
    
    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the A* Grid.
        
        Args:
            grid: A 2D list representing the map. 
                  Values > 0 indicate walkable cells (weight/cost).
                  Value 0 indicates a wall (unwalkable).
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
            
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self._directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path between start and end coordinates using A*.
        
        Args:
            start: Tuple of (row, col) indicating the starting position.
            end: Tuple of (row, col) indicating the target position.
            
        Returns:
            A list of (row, col) tuples representing the path from start to end.
            Returns None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of grid bounds.
        """
        # Validate bounds
        sr, sc = start
        er, ec = end
        
        self._validate_coordinates(start)
        self._validate_coordinates(end)
        
        # Handle start == end case
        if start == end:
            return [start]
        
        # Check if start or end are walls (0)
        if self.grid[sr][sc] == 0 or self.grid[er][ec] == 0:
            # Cannot start on a wall or reach a target on a wall
            return None

        open_set: List[Tuple[float, int, int, int]] = []
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0}
        f_score: Dict[Tuple[int, int], float] = {start: self._heuristic(start, end)}
        
        # Push (f_score, counter, row, col) to heapq
        # Counter ensures FIFO ordering when f_scores are equal (tie-breaking)
        counter = 0
        heapq.heappush(open_set, (f_score[start], counter, sr, sc))
        
        while open_set:
            _, _, curr_r, curr_c = heapq.heappop(open_set)
            current = (curr_r, curr_c)
            
            if current == end:
                return self._reconstruct_path(came_from, current)
            
            # Neighbors
            for dr, dc in self._directions:
                nr, nc = curr_r + dr, curr_c + dc
                neighbor = (nr, nc)
                
                if not self._is_valid_coordinate(neighbor):
                    continue
                
                # Wall check (value 0)
                if self.grid[nr][nc] == 0:
                    continue
                
                # Calculate tentative cost
                # Cost to move is determined by the destination cell's weight
                tentative_g = g_score.get(current, float('inf')) + self.grid[nr][nc]
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    new_f = tentative_g + self._heuristic(neighbor, end)
                    f_score[neighbor] = new_f
                    counter += 1
                    heapq.heappush(open_set, (new_f, counter, nr, nc))
                    
        return None

    def _validate_coordinates(self, pos: Tuple[int, int]) -> None:
        if pos is None:
            raise ValueError("Position cannot be None.")
        r, c = pos
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(f"Coordinates {pos} are out of bounds.")

    def _is_valid_coordinate(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _heuristic(self, node: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """
        Calculates the Manhattan distance between two points.
        Admissible heuristic for 4-directional grid movement.
        """
        r1, c1 = node
        r2, c2 = goal
        return abs(r1 - r2) + abs(c1 - c2)

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

import pytest

def test_basic_path_exists():
    """Tests finding a simple path in an open grid."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    finder = AStarGrid(grid)
    path = finder.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) >= 5  # Minimum Manhattan distance path length is 5 nodes (0,0 to 2,2)

def test_start_equals_end():
    """Tests that finding path from start to itself returns immediate result."""
    grid = [[1, 2], [3, 4]]
    finder = AStarGrid(grid)
    path = finder.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_unreachable_by_walls():
    """Tests that path returns None when target is surrounded by walls (0)."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    finder = AStarGrid(grid)
    path = finder.find_path((0, 0), (2, 2))
    assert path is None

def test_start_out_of_bounds():
    """Tests that start coordinates out of range raise ValueError."""
    grid = [[1]]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError):
        finder.find_path((5, 5), (0, 0))

def test_end_out_of_bounds():
    """Tests that end coordinates out of range raise ValueError."""
    grid = [[1]]
    finder = AStarGrid(grid)
    with pytest.raises(ValueError):
        finder.find_path((0, 0), (-1, -1))

def test_weighted_optimality():
    """Tests that A* chooses the cheaper path even if slightly longer in steps."""
    # Two paths to (0, 2): 
    # Top path costs 10 + 10 = 20
    # Bottom path costs 1 + 1 = 2 (but takes more steps vertically)
    grid = [
        [1, 10, 10], # High cost top row
        [1,  1,  1], # Low cost bottom row
        [1,  1,  1]  # Return path
    ]
    # Start (0,0), End (0,2)
    # Path 1: (0,0)->(0,1)->(0,2). Cost: 10 + 10 = 20.
    # Path 2: (0,0)->(1,0)->(1,1)->(1,2)->(0,2). Cost: 1 + 1 + 1 + 10 = 13.
    # Wait, End is (0,2) which is 10. 
    # Path 2 cost to enter (0,2) is 10 regardless. 
    # Total Path 1: 10 + 10 = 20.
    # Total Path 2: 1 (to 1,0) + 1 (to 1,1) + 1 (to 1,2) + 10 (to 0,2) = 13.
    # A* should pick Path 2.
    
    finder = AStarGrid(grid)
    path = finder.find_path((0, 0), (0, 2))
    assert path is not None
    # Calculate actual cost of found path
    total_cost = 0
    for i, (r, c) in enumerate(path):
        if i == 0:
            total_cost += 0 # Start cell cost usually 0 in traversal logic, but depends on implementation. 
            # My implementation sums grid[nr][nc] when moving TO neighbor.
            continue
        prev_r, prev_c = path[i-1]
        # Cost is added when entering new cell
        total_cost += grid[r][c]
        
    assert total_cost < 20 # Should definitely choose the lower weighted path
    assert (1, 0) in path # Should deviate down into low cost area