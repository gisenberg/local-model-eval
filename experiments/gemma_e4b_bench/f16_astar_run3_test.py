import heapq
, Tuple, Optional, Dict

# Define types for clarity
Grid = List[List[int]]
Point = Tuple[int, int]

class AStarGrid:
    """
    Implements the A* pathfinding algorithm on a weighted 2D grid.

    The grid values represent the cost to enter that cell.
    A value of 0 represents a wall (impassable).
    """

    def __init__(self, grid: Grid):
        """
        Initializes the AStarGrid with the map.

        Args:
            grid: A 2D list where each element is the movement cost.
                  0 indicates a wall.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, point: Point) -> bool:
        """Checks if a point is within bounds and not a wall."""
        r, c = point
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] > 0

    def _heuristic(self, a: Point, b: Point) -> float:
        """
        Calculates the Manhattan distance heuristic (h(n)).
        h(n) = |x1 - x2| + |y1 - y2|
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Point, end: Point) -> Optional[List[Point]]:
        """
        Finds the shortest path from start to end using A*.

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
        
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError(f"Start point {start} is a wall.")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError(f"End point {end} is a wall.")

        if start == end:
            return [start]

        # Priority queue: (f_score, g_score, point)
        # We include g_score to break ties consistently, though not strictly necessary for correctness.
        open_list: List[Tuple[float, float, Point]] = []
        
        # g_score: Cost from start to current node
        g_score: Dict[Point, float] = {start: 0.0}
        
        # f_score: Estimated total cost (g_score + heuristic)
        f_score: Dict[Point, float] = {start: self._heuristic(start, end)}
        
        # came_from: Dictionary to reconstruct the path
        came_from: Dict[Point, Point] = {}

        # Initialize the heap with the start node
        heapq.heappush(open_list, (f_score[start], 0.0, start))

        while open_list:
            # Pop the node with the lowest f_score
            current_f, current_g, current = heapq.heappop(open_list)

            if current == end:
                # Path found, reconstruct and return
                path: List[Point] = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            # Explore neighbors (4-directional movement)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not self._is_valid(neighbor):
                    continue

                # Cost to move to the neighbor is the weight of the neighbor cell
                movement_cost = self.grid[neighbor[0]][neighbor[1]]
                
                # Tentative g_score is the cost to reach the neighbor through the current node
                tentative_g_score = g_score[current] + movement_cost

                # If this path to the neighbor is better than any previously found path
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    # This path is the best so far. Record it.
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    
                    h_score = self._heuristic(neighbor, end)
                    f_score[neighbor] = tentative_g_score + h_score
                    
                    # Add or update the neighbor in the priority queue
                    # We push the new, better score; the old, worse entry will be ignored later.
                    heapq.heappush(open_list, (f_score[neighbor], tentative_g_score, neighbor))

        # Open list is empty and the end was not reached
        return None

if __name__ == '__main__':
    # Example Usage
    # Grid: 1=cost 1, 5=cost 5, 0=wall
    test_grid = [
        [1, 1, 5, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1],
        [5, 1, 1, 1]
    ]
    
    astar = AStarGrid(test_grid)
    start_point = (0, 0)
    end_point = (3, 3)

    print(f"Finding path from {start_point} to {end_point}...")
    path = astar.find_path(start_point, end_point)

    if path:
        print("\nPath found:")
        print(path)
        
        # Calculate total cost for verification
        total_cost = 0
        for i in range(1, len(path)):
            r, c = path[i]
            total_cost += test_grid[r][c]
        print(f"Total path cost: {total_cost}")
    else:
        print("\nNo path found.")

import pytest
, Point, Grid
, Tuple

# --- Fixtures ---

@pytest.fixture
def simple_grid() -> Grid:
    """A basic grid with varying weights and one wall."""
    # Costs: 1, 2, 1
    #        1, 0, 1
    #        1, 1, 1
    return [
        [1, 2, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]

@pytest.fixture
def complex_grid() -> Grid:
    """A larger grid for complex pathfinding."""
    return [
        [1, 1, 1, 1, 1],
        [1, 5, 0, 5, 1],
        [1, 1, 1, 1, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ]

# --- Test Cases ---

def test_start_equals_end(simple_grid) -> None:
    """Test case where the start and end points are the same."""
    grid = AStarGrid(simple_grid)
    start = (0, 0)
    end = (0, 0)
    path = grid.find_path(start, end)
    assert path == [start]

def test_path_found_simple(simple_grid) -> None:
    """Test finding a path on a simple grid."""
    grid = AStarGrid(simple_grid)
    start = (0, 0)
    end = (2, 2)
    path = grid.find_path(start, end)
    
    assert path is not None
    assert len(path) > 0
    assert path[0] == start
    assert path[-1] == end

def test_no_path_blocked(simple_grid) -> None:
    """Test case where the destination is completely walled off."""
    # Modify grid to completely block the path to (2, 2)
    blocked_grid = [
        [1, 1, 1],
        [1, 0, 0],
        [1, 0, 1]
    ]
    grid = AStarGrid(blocked_grid)
    start = (0, 0)
    end = (2, 2)
    path = grid.find_path(start, end)
    assert path is None

def test_optimal_path_with_weights(complex_grid) -> None:
    """Test that A* chooses the path with the lowest total weight."""
    grid = AStarGrid(complex_grid)
    start = (0, 0)
    end = (4, 4)
    path = grid.find_path(start, end)
    
    assert path is not None
    
    # Verify optimality by calculating the cost of the found path
    total_cost = 0
    for i in range(1, len(path)):
        r, c = path[i]
        total_cost += grid.grid[r][c]
    
    # The expected optimal path avoids the high-cost '5' cells and the '0' walls.
    # A known optimal path cost for this setup is 10.
    assert total_cost == 10

def test_out_of_bounds_error(simple_grid) -> None:
    """Test handling of start or end points outside the grid boundaries."""
    grid = AStarGrid(simple_grid)
    
    # Test start out of bounds
    with pytest.raises(ValueError, match="Start point"):
        grid.find_path((3, 0), (0, 0))
        
    # Test end out of bounds
    with pytest.raises(ValueError, match="End point"):
        grid.find_path((0, 0), (0, 3))

def test_wall_start_or_end_error(simple_grid) -> None:
    """Test handling when start or end points are walls (cost 0)."""
    grid = AStarGrid(simple_grid)
    
    # Test start is a wall (0, 1) in the simple grid is 2, so let's use a wall cell
    # In simple_grid, (1, 1) is the wall
    with pytest.raises(ValueError, match="Start point"):
        grid.find_path((1, 1), (2, 2))
        
    # Test end is a wall
    with pytest.raises(ValueError, match="End point"):
        grid.find_path((0, 0), (1, 1))

def test_empty_grid_initialization() -> None:
    """Test initialization failure for an empty grid."""
    with pytest.raises(ValueError, match="Grid cannot be empty"):
        AStarGrid([])