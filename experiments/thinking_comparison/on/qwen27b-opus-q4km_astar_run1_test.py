import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid for pathfinding.
        grid: 2D list where 0 represents a wall and positive integers represent movement cost.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        # Directions: Up, Down, Left, Right
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _get_neighbors(self, r: int, c: int) -> List[Tuple[int, int, int]]:
        """
        Get valid neighbors (row, col, cost) for a given cell.
        Returns a list of tuples containing neighbor coordinates and the cost to enter them.
        """
        neighbors = []
        for dr, dc in self.directions:
            nr, nc = r + dr, c + dc
            if self._is_valid(nr, nc):
                cost = self.grid[nr][nc]
                if cost > 0:  # Only add if not a wall (0)
                    neighbors.append((nr, nc, cost))
        return neighbors

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance heuristic between two points.
        h(n) = |x1 - x2| + |y1 - y2|
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        Returns a list of coordinates (row, col) or None if no path exists.
        Raises ValueError if start or end are out of bounds.
        """
        # 1. Validate inputs
        if not self._is_valid(start[0], start[1]):
            raise ValueError(f"Start position {start} is out of bounds.")
        if not self._is_valid(end[0], end[1]):
            raise ValueError(f"End position {end} is out of bounds.")

        # 2. Handle edge cases
        if start == end:
            return [start]
        
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # 3. Initialize A* structures
        # Open set: min-heap storing (f_score, g_score, row, col)
        # We use g_score as a tie-breaker in the heap tuple to ensure stability
        open_set = [(0, 0, start[0], start[1])]
        
        # Track visited nodes to avoid cycles and redundant processing
        closed_set = set()
        
        # Track g_scores to find the optimal path
        g_scores = {start: 0}
        
        # Track parents for path reconstruction
        came_from = {}

        while open_set:
            # Pop node with lowest f_score
            _, current_g, current_r, current_c = heapq.heappop(open_set)
            current = (current_r, current_c)

            # If we reached the goal
            if current == end:
                return self._reconstruct_path(came_from, start, end)

            # Add to closed set
            closed_set.add(current)

            # Explore neighbors
            for nr, nc, move_cost in self._get_neighbors(current_r, current_c):
                neighbor = (nr, nc)

                if neighbor in closed_set:
                    continue

                # Calculate tentative g_score
                tentative_g = current_g + move_cost

                # If this path to neighbor is better than any previous one, record it
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    
                    # Push to open set
                    heapq.heappush(open_set, (f_score, tentative_g, nr, nc))
                    came_from[neighbor] = current

        return None  # No path found

    def _reconstruct_path(self, came_from: dict, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct the path from end to start using the came_from dictionary."""
        path = [end]
        current = end
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

import pytest


def calculate_path_cost(grid, path):
    """Helper to calculate total cost of a path for verification."""
    if not path:
        return 0
    total = 0
    for r, c in path:
        total += grid[r][c]
    return total

def test_simple_path_uniform_grid():
    """Test finding a simple path on a uniform grid."""
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
    # Length should be 5 steps (Manhattan distance + 1)
    assert len(path) == 5
    assert calculate_path_cost(grid, path) == 5

def test_path_around_obstacles():
    """Test finding a path that goes around walls."""
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 3))
    
    assert path is not None
    # Direct path blocked, must go down and around
    assert (1, 0) in path or (1, 3) in path # Must touch row 1
    assert calculate_path_cost(grid, path) == 5

def test_weighted_grid_optimality():
    """Test that the algorithm prefers lower-cost cells over shorter distance."""
    # A path going straight right costs 10+10+10 = 30
    # A path going down then right costs 1+1+1+1 = 4
    grid = [
        [1, 10, 10, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 3))
    
    assert path is not None
    # The optimal path goes down to row 1, across, then up
    # Cost: 1(start) + 1(down) + 1(right) + 1(right) + 1(up) = 5
    # Wait, start is (0,0). End is (0,3).
    # Path 1 (Top): (0,0)->(0,1)->(0,2)->(0,3) Cost: 1+10+10+1 = 22
    # Path 2 (Bottom): (0,0)->(1,0)->(1,1)->(1,2)->(1,3)->(0,3) Cost: 1+1+1+1+1+1 = 6
    assert calculate_path_cost(grid, path) == 6

def test_no_path_exists():
    """Test that None is returned when the destination is unreachable."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

def test_start_equals_end():
    """Test edge case where start and end are the same."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    assert path == [(1, 1)]

def test_invalid_coordinates():
    """Test that ValueError is raised for out-of-bounds coordinates."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
    
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))

if __name__ == "__main__":
    pytest.main([__file__])