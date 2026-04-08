from typing import List, Tuple, Optional
import heapq

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid with the given movement cost grid.
        
        Args:
            grid: 2D list where each cell represents movement cost.
                  0 = impassable wall, positive int = cost to enter.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting coordinate (row, col)
            end: Ending coordinate (row, col)
        
        Returns:
            List of coordinates from start to end, or None if no path exists.
        
        Raises:
            ValueError: If start or end is out of bounds.
        """
        # Edge case: start equals end
        if start == end:
            return [start]
        
        # Check if start or end is out of bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End is out of bounds")
        
        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None
        
        # Initialize A* algorithm
        open_set = []
        heapq.heappush(open_set, (0, 0, start))  # (f, g, node)
        g_scores = {start: 0}
        parent = {}
        closed_set = set()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while open_set:
            f, g, current = heapq.heappop(open_set)
            
            # Check if current node is the end
            if current == end:
                # Reconstruct path
                path = []
                node = current
                while node in parent:
                    path.append(node)
                    node = parent[node]
                path.append(start)
                path.reverse()
                return path
            
            # Skip if node is already processed
            if current in closed_set:
                continue
            
            # Check if current g matches stored g_scores
            if g_scores.get(current, float('inf')) != g:
                continue
            
            closed_set.add(current)
            
            # Explore neighbors
            for dr, dc in directions:
                nr, nc = current[0] + dr, current[1] + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.grid[nr][nc] == 0:
                        continue  # Skip walls
                    
                    tentative_g = g + self.grid[nr][nc]
                    if (nr, nc) not in g_scores or tentative_g < g_scores[(nr, nc)]:
                        g_scores[(nr, nc)] = tentative_g
                        h = abs(nr - end[0]) + abs(nc - end[1])
                        f = tentative_g + h
                        heapq.heappush(open_set, (f, tentative_g, (nr, nc)))
                        parent[(nr, nc)] = current
        
        # No path found
        return None

# Pytest tests
import pytest

def test_simple_path():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5  # Path length is 5 steps (4 moves)
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

def test_path_around_obstacles():
    grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert len(path) == 5  # Path length is 5 steps
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)

def test_weighted_grid():
    grid = [[1, 10, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    total_cost = sum(grid[r][c] for r, c in path)
    assert total_cost == 4  # Minimal cost path through the bottom row

def test_no_path():
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_start_equals_end():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_invalid_coordinates():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((3, 0), (2, 2))