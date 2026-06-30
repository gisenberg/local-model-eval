import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* Pathfinding implementation on a weighted 2D grid.
    
    The grid is represented as a list of lists of integers.
    - 0 represents a wall (impassable).
    - Positive integers represent the cost to enter/traverse the cell.
    """
    def __init__(self, grid: List[List[int]]):
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if coordinates are within grid bounds."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid, non-wall neighbors (4-directional)."""
        r, c = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self._is_valid((nr, nc)):
                # 0 represents a wall
                if self.grid[nr][nc] != 0:
                    neighbors.append((nr, nc))
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the optimal path from start to end using A* algorithm.
        
        Args:
            start: Tuple (row, col) of starting position.
            end: Tuple (row, col) of ending position.
            
        Returns:
            List of tuples representing the path from start to end, or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # 1. Validation
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end coordinates are out of bounds.")
        
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # 2. Optimization: Start equals End
        if start == end:
            return [start]

        # 3. A* Initialization
        # Priority Queue: (f_score, g_score, position)
        open_set = []
        start_g = 0
        start_f = self._heuristic(start, end)
        heapq.heappush(open_set, (start_f, start_g, start))
        
        # g_scores map: position -> lowest cost found so far
        g_scores = {start: 0}
        
        # parents map: position -> parent position
        parents = {}
        
        while open_set:
            # Pop node with lowest f_score
            current_f, current_g, current = heapq.heappop(open_set)
            
            # If we found a better path to this node already, skip
            if current_g > g_scores.get(current, float('inf')):
                continue
            
            # Goal check
            if current == end:
                return self._reconstruct_path(parents, current)
            
            # Explore neighbors
            for neighbor in self._get_neighbors(current):
                # Cost to move to neighbor is the weight of the neighbor cell
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = current_g + move_cost
                
                # If this path is better than any previous path to neighbor
                if tentative_g < g_scores.get(neighbor, float('inf')):
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    parents[neighbor] = current
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
        
        return None # No path found

    def _reconstruct_path(self, parents: dict, end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from parents map."""
        path = [end]
        current = end
        while current in parents:
            current = parents[current]
            path.append(current)
        path.reverse()
        return path

import pytest
from typing import List, Tuple

# Assuming AStarGrid is imported or defined above
# 
def test_simple_path():
    """Test basic pathfinding on a grid without walls."""
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
    # Path length should be 5 (Manhattan distance 4 steps + 1 start node)
    assert len(path) == 5

def test_wall_avoidance():
    """Test that pathfinding correctly avoids walls (0)."""
    grid = [
        [1, 0, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Start (0,0), End (0,2). Direct path blocked by wall at (0,1).
    path = astar.find_path((0, 0), (0, 2))
    
    assert path is not None
    # Path must go down to row 1 to bypass wall
    assert (0, 1) not in path
    assert (1, 1) in path or (1, 0) in path or (1, 2) in path

def test_start_equals_end():
    """Test handling when start and end are the same."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    
    assert path == [(0, 0)]

def test_no_path():
    """Test returning None when no path exists."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    # Start (0,0), End (2,2). Surrounded by walls.
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is None

def test_out_of_bounds():
    """Test ValueError for out-of-bounds coordinates."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2)) # Row 2 is out of bounds

def test_weighted_path_optimality():
    """Test that A* chooses the path with lower total weight."""
    # Grid with high cost cells (10) and low cost cells (1)
    # 1  10  1
    # 1   1  1
    # 1  10  1
    grid = [
        [1, 10, 1],
        [1, 1, 1],
        [1, 10, 1]
    ]
    astar = AStarGrid(grid)
    # Start (0,0), End (2,2)
    # Optimal path goes through center (1,1) cost 1.
    # Paths through (0,1) or (2,1) cost 10.
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    # The path should NOT contain the high-cost cells (0,1) or (2,1)
    assert (0, 1) not in path
    assert (2, 1) not in path
    # The path SHOULD contain the low-cost center cell (1,1)
    assert (1, 1) in path