import heapq
, Tuple, Optional, Dict, Set

# Define types for clarity
Grid = List[List[float]]
Point = Tuple[int, int]

class AStarGrid:
    """
    Implements the A* pathfinding algorithm on a weighted 2D grid.

    The grid values represent the cost to enter that cell.
    A value of 0 represents a wall (impassable).
    """

    def __init__(self, grid: Grid):
        """
        Initializes the AStarGrid.

        Args:
            grid: A 2D list of floats where each float is the movement cost.
                  0.0 indicates a wall.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _heuristic(self, a: Point, b: Point) -> float:
        """
        Calculates the Manhattan distance heuristic between two points.
        h(n) = |x1 - x2| + |y1 - y2|
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, current: Point) -> List[Point]:
        """
        Returns the 4-directional neighbors of a given point.
        """
        r, c = current
        # (dr, dc) for up, down, left, right
        neighbors = [
            (r - 1, c),  # Up
            (r + 1, c),  # Down
            (r, c - 1),  # Left
            (r, c + 1)   # Right
        ]
        return neighbors

    def find_path(self, start: Point, end: Point) -> Optional[List[Point]]:
        """
        Finds the optimal path from start to end using A*.

        Args:
            start: The starting coordinates (row, col).
            end: The target coordinates (row, col).

        Returns:
            A list of points representing the optimal path, or None if no path exists.

        Raises:
            ValueError: If start or end points are out of bounds or are walls.
        """
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start point {start} is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End point {end} is out of bounds.")
        
        if self.grid[start[0]][start[1]] == 0.0:
            raise ValueError(f"Start point {start} is a wall.")
        if self.grid[end[0]][end[1]] == 0.0:
            raise ValueError(f"End point {end} is a wall.")

        if start == end:
            return [start]

        # Priority queue: stores (f_score, g_score, point)
        # We include g_score to break ties consistently, though f_score is primary.
        open_set: List[Tuple[float, float, Point]] = []
        heapq.heappush(open_set, (0.0 + self._heuristic(start, end), 0.0, start))

        # g_score: Cost from start to the current node
        g_score: Dict[Point, float] = {start: 0.0}
        
        # came_from: Used to reconstruct the path
        came_from: Dict[Point, Point] = {}

        while open_set:
            # Pop the node with the lowest f_score
            _, current_g, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Return reversed path (start to end)

            for neighbor in self._get_neighbors(current):
                r, c = neighbor
                
                # 1. Check bounds
                if not (0 <= r < self.rows and 0 <= c < self.cols):
                    continue
                
                # 2. Check for walls (cost 0)
                cost_to_enter = self.grid[r][c]
                if cost_to_enter == 0.0:
                    continue

                # Tentative g_score is the cost to reach the neighbor through the current node
                tentative_g_score = current_g + cost_to_enter

                # If this path to the neighbor is better than any previous one found
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    
                    # Update records
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    
                    f_score = tentative_g_score + self._heuristic(neighbor, end)
                    
                    # Push to priority queue
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

        # If the loop finishes without reaching the end
        return None

if __name__ == '__main__':
    # Example Usage:
    # Grid: 3x3
    # Costs: 1.0 (normal), 5.0 (expensive), 0.0 (wall)
    test_grid = [
        [1.0, 1.0, 1.0],
        [1.0, 0.0, 5.0],
        [1.0, 1.0, 1.0]
    ]
    
    astar = AStarGrid(test_grid)
    start_point = (0, 0)
    end_point = (2, 2)

    print(f"Finding path from {start_point} to {end_point}...")
    path = astar.find_path(start_point, end_point)

    if path:
        print("Path found:", path)
        # Calculate total cost for verification
        total_cost = 0.0
        for i in range(len(path) - 1):
            r, c = path[i+1]
            total_cost += test_grid[r][c]
        print(f"Total path cost: {total_cost}")
    else:
        print("No path found.")

import pytest
, Tuple
, Grid, Point

# --- Test Fixtures ---

@pytest.fixture
def simple_grid() -> Grid:
    """A simple 3x3 grid with uniform cost 1.0."""
    return [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ]

@pytest.fixture
def weighted_grid() -> Grid:
    """A 4x4 grid with varying costs and a wall."""
    # Costs: 1.0 (normal), 5.0 (expensive), 0.0 (wall)
    return [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 5.0, 0.0, 1.0],  # Wall at (1, 2)
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0]
    ]

@pytest.fixture
def blocked_grid() -> Grid:
    """A grid where the path is completely blocked."""
    return [
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 0.0, 1.0]
    ]

# --- Test Cases ---

def test_start_equals_end(simple_grid: Grid):
    """Test case where the start and end points are the same."""
    astar = AStarGrid(simple_grid)
    start = (1, 1)
    end = (1, 1)
    path = astar.find_path(start, end)
    assert path == [start]

def test_simple_path_found(simple_grid: Grid):
    """Test finding a straightforward path in a uniform grid."""
    astar = AStarGrid(simple_grid)
    start = (0, 0)
    end = (2, 2)
    path = astar.find_path(start, end)
    
    assert path is not None
    assert len(path) > 1
    assert path[0] == start
    assert path[-1] == end

def test_weighted_path_optimality(weighted_grid: Grid):
    """Test that A* chooses the lowest cost path, avoiding the expensive cell (1, 1)."""
    astar = AStarGrid(weighted_grid)
    start = (0, 0)
    end = (3, 3)
    path = astar.find_path(start, end)
    
    assert path is not None
    
    # The path should avoid (1, 1) which costs 5.0, favoring the route around the wall.
    # A direct path through (1, 1) would cost significantly more.
    assert (1, 1) not in path
    assert path[-1] == end

def test_path_blocked_no_path(blocked_grid: Grid):
    """Test case where the destination is unreachable due to walls."""
    astar = AStarGrid(blocked_grid)
    start = (0, 0)
    end = (0, 2) # Blocked by (0, 1)
    path = astar.find_path(start, end)
    assert path is None

def test_out_of_bounds_start(simple_grid: Grid):
    """Test handling of start point outside grid boundaries."""
    astar = AStarGrid(simple_grid)
    with pytest.raises(ValueError, match="Start point"):
        astar.find_path((3, 0), (1, 1))

def test_wall_start_or_end(weighted_grid: Grid):
    """Test handling when the start or end point is a wall (cost 0)."""
    astar = AStarGrid(weighted_grid)
    wall_point = (1, 2) # A wall in the weighted grid
    
    # Test start is a wall
    with pytest.raises(ValueError, match="Start point"):
        astar.find_path(wall_point, (3, 3))
        
    # Test end is a wall
    with pytest.raises(ValueError, match="End point"):
        astar.find_path((0, 0), wall_point)

def test_grid_initialization_error():
    """Test initialization failure for an empty grid."""
    with pytest.raises(ValueError, match="Grid cannot be empty"):
        AStarGrid([])