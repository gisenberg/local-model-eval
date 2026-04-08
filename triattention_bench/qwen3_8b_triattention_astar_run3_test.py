import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid with the given movement cost grid.
        
        Args:
            grid: A 2D list where each cell represents the movement cost (0 = impassable wall, positive int = cost to enter).
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Tuple (row, col) representing the starting position.
            end: Tuple (row, col) representing the ending position.
        
        Returns:
            List of (row, col) coordinates representing the path from start to end, or None if no path exists.
        
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        start_row, start_col = start
        end_row, end_col = end

        # Check if start or end is out of bounds
        if start_row < 0 or start_row >= self.rows or start_col < 0 or start_col >= self.cols:
            raise ValueError("Start coordinates are out of bounds")
        if end_row < 0 or end_row >= self.rows or end_col < 0 or end_col >= self.cols:
            raise ValueError("End coordinates are out of bounds")

        # Check if start or end is a wall
        if self.grid[start_row][start_col] == 0:
            return None
        if self.grid[end_row][end_col] == 0:
            return None

        # Check if start == end
        if start == end:
            return [start]

        # Initialize open set
        open_set = []
        g_scores = {}
        parents = {}
        visited = set()

        # Start node's g is grid value
        start_g = self.grid[start_row][start_col]
        g_scores[(start_row, start_col)] = start_g
        parents[(start_row, start_col)] = None

        # Heuristic for start node
        h_start = abs(start_row - end_row) + abs(start_col - end_col)
        f_start = start_g + h_start
        heapq.heappush(open_set, (f_start, start_row, start_col))

        while open_set:
            current_f, current_row, current_col = heapq.heappop(open_set)
            current_node = (current_row, current_col)

            if current_node in visited:
                continue
            visited.add(current_node)

            # Check if current node is end
            if current_node == (end_row, end_col):
                # Reconstruct path
                path = []
                current = current_node
                while current is not None:
                    path.append(current)
                    current = parents.get(current, None)
                path.reverse()
                return path

            # Explore neighbors
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in directions:
                new_row = current_row + dr
                new_col = current_col + dc

                # Check if new_row and new_col are within bounds
                if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                    # Check if it's a wall
                    if self.grid[new_row][new_col] == 0:
                        continue

                    # Calculate tentative_g
                    tentative_g = g_scores[current_node] + self.grid[new_row][new_col]
                    new_node = (new_row, new_col)

                    # Check if new_node is not in g_scores or tentative_g is better
                    if new_node not in g_scores or tentative_g < g_scores[new_node]:
                        g_scores[new_node] = tentative_g
                        parents[new_node] = current_node
                        h_new = abs(new_row - end_row) + abs(new_col - end_col)
                        f_new = tentative_g + h_new
                        heapq.heappush(open_set, (f_new, new_row, new_col))

        # If end not found
        return None

# Test cases
import pytest

def test_simple_path():
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0,0), (2,2))
    assert path == [(0,0), (0,1), (0,2), (1,2), (2,2)]

def test_path_around_obstacles():
    grid = [[1, 1, 1, 1], [1, 0, 0, 1], [1, 1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0,0), (2,3))
    assert path == [(0,0), (1,0), (2,0), (2,1), (2,2), (2,3)]

def test_weighted_grid():
    grid = [[1, 3, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0,0), (2,2))
    assert path == [(0,0), (1,0), (2,0), (2,1), (2,2)]

def test_no_path():
    grid = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    astar = AStarGrid(grid)
    path = astar.find_path((0,0), (2,2))
    assert path is None

def test_start_equals_end():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0,0), (0,0))
    assert path == [(0,0)]

def test_invalid_coordinates():
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((2,0), (0,0))