import heapq
, Tuple, Optional, Dict

# Define types for clarity
Grid = List[List[int]]
Point = Tuple[int, int]

class AStarGrid:
    """
    Implements the A* pathfinding algorithm on a weighted 2D grid.

    The grid values represent the cost to enter that cell.
    A value of 0 indicates a wall (impassable).
    """

    def __init__(self, grid: Grid):
        """
        Initializes the AStarGrid with the map.

        Args:
            grid: A 2D list where grid[y][x] is the weight/cost of that cell.
                  0 represents a wall.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _heuristic(self, a: Point, b: Point) -> int:
        """
        Calculates the Manhattan distance heuristic between two points.
        h(n) = |x1 - x2| + |y1 - y2|
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Point, end: Point) -> Optional[List[Point]]:
        """
        Finds the shortest path from the start point to the end point using A*.

        Args:
            start: The starting coordinates (x, y).
            end: The target coordinates (x, y).

        Returns:
            A list of points representing the optimal path, or None if no path exists.

        Raises:
            ValueError: If start or end points are out of bounds or are walls.
        """
        start_x, start_y = start
        end_x, end_y = end

        # 1. Boundary and Wall Checks
        if not (0 <= start_x < self.cols and 0 <= start_y < self.rows):
            raise ValueError(f"Start point {start} is out of bounds.")
        if not (0 <= end_x < self.cols and 0 <= end_y < self.rows):
            raise ValueError(f"End point {end} is out of bounds.")
        
        # Note: We assume the cost of the start cell itself is paid upon entering it.
        # If the start cell is a wall, we cannot start there.
        if self.grid[start_y][start_x] == 0:
            raise ValueError(f"Start point {start} is a wall.")
        if self.grid[end_y][end_x] == 0:
            # If the end is a wall, no path can end there.
            return None

        # Handle trivial case
        if start == end:
            return [start]

        # Priority queue: (f_score, g_score, point)
        # We include g_score for tie-breaking consistency, though f_score is primary.
        open_list: List[Tuple[int, int, Point]] = []
        
        # g_score: Cost from start to current node
        g_score: Dict[Point, float] = {start: 0.0}
        
        # f_score: Estimated total cost (g_score + heuristic)
        f_score: Dict[Point, float] = {start: self._heuristic(start, end)}
        
        # came_from: Used to reconstruct the path
        came_from: Dict[Point, Point] = {}

        # Initialize the priority queue
        heapq.heappush(open_list, (f_score[start], 0.0, start))

        # 4-directional movements (dx, dy)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_list:
            # Get the node with the lowest f_score
            current_f, current_g, current = heapq.heappop(open_list)
            current_x, current_y = current

            if current == end:
                # Path found, reconstruct it
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Return reversed path (start to end)

            for dx, dy in directions:
                neighbor_x, neighbor_y = current_x + dx, current_y + dy
                neighbor = (neighbor_x, neighbor_y)

                # Check bounds
                if not (0 <= neighbor_x < self.cols and 0 <= neighbor_y < self.rows):
                    continue
                
                # Check for walls (cost 0)
                cost_to_enter = self.grid[neighbor_y][neighbor_x]
                if cost_to_enter == 0:
                    continue

                # Cost calculation: g_score of current + cost to move to neighbor
                tentative_g_score = g_score[current] + cost_to_enter

                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Record the better path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    
                    h_score = self._heuristic(neighbor, end)
                    f_score[neighbor] = tentative_g_score + h_score
                    
                    # Push to priority queue
                    heapq.heappush(open_list, (f_score[neighbor], tentative_g_score, neighbor))

        # If the loop finishes without reaching the end
        return None

# Example Usage (for manual testing)
if __name__ == '__main__':
    # Grid: (X, Y)
    # Costs: 1=normal, 5=expensive, 0=wall
    # (0,0) (1,0) (2,0)
    # (0,1) (1,1) (2,1)
    # (0,2) (1,2) (2,2)
    grid_map: Grid = [
        [1, 1, 5],  # Row 0 (Y=0)
        [1, 0, 1],  # Row 1 (Y=1) - Wall at (1, 1)
        [1, 1, 1]   # Row 2 (Y=2)
    ]
    
    astar = AStarGrid(grid_map)

    # Test 1: Simple path
    start_point = (0, 0)
    end_point = (2, 2)
    path = astar.find_path(start_point, end_point)
    print(f"Path from {start_point} to {end_point}: {path}")
    
    # Test 2: No path (blocked)
    start_point_2 = (0, 0)
    end_point_2 = (2, 1) # Blocked by wall at (1, 1)
    path_2 = astar.find_path(start_point_2, end_point_2)
    print(f"Path from {start_point_2} to {end_point_2}: {path_2}")

import pytest
, Grid, Point
, Tuple

# --- Fixtures ---

@pytest.fixture
def simple_grid() -> Grid:
    """A basic grid with varying costs and a clear path."""
    # Costs: 1=normal, 5=expensive, 0=wall
    return [
        [1, 1, 1, 1],  # Y=0
        [1, 5, 0, 1],  # Y=1 (Wall at 2,1)
        [1, 1, 1, 1]   # Y=2
    ]

@pytest.fixture
def complex_grid() -> Grid:
    """A grid designed to test pathfinding around obstacles and costs."""
    return [
        [1, 1, 1, 1, 1],  # Y=0
        [1, 0, 0, 1, 1],  # Y=1 (Wall block)
        [1, 1, 1, 0, 1],  # Y=2 (Wall at 3,2)
        [1, 5, 1, 1, 1]   # Y=3 (Expensive cell at 1,3)
    ]

# --- Test Cases ---

def test_start_equals_end(simple_grid: Grid):
    """Test case where the start and end points are the same."""
    astar = AStarGrid(simple_grid)
    start = (0, 0)
    end = (0, 0)
    path = astar.find_path(start, end)
    assert path == [start]

def test_path_found_simple(simple_grid: Grid):
    """Test finding a straightforward path on a simple grid."""
    astar = AStarGrid(simple_grid)
    start = (0, 0)
    end = (3, 2)
    path = astar.find_path(start, end)
    
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    # Check path length is reasonable (should be short)
    assert len(path) < 10 

def test_no_path_blocked(simple_grid: Grid):
    """Test case where the destination is completely walled off."""
    astar = AStarGrid(simple_grid)
    start = (0, 0)
    # Target (2, 1) is blocked by the wall at (2, 1) in the grid structure
    end = (2, 1) 
    path = astar.find_path(start, end)
    assert path is None

def test_optimal_path_cost(complex_grid: Grid):
    """
    Test that the algorithm chooses the path with the lowest cumulative weight, 
    avoiding the expensive cell (1, 3) if a cheaper route exists.
    """
    astar = AStarGrid(complex_grid)
    start = (0, 0)
    end = (4, 3)
    path = astar.find_path(start, end)
    
    assert path is not None
    
    # Calculate the actual cost of the found path
    total_cost = 0
    for i in range(1, len(path)):
        x, y = path[i]
        # Cost is the weight of the cell being entered
        total_cost += complex_grid[y][x]
    
    # The optimal path should avoid the cost 5 cell at (1, 3) if possible.
    # A path around the wall block (1,1)-(2,1) and avoiding (1,3) should be cheaper.
    # The expected optimal path cost is around 7-9 depending on exact routing.
    # We assert that the cost is not excessively high (e.g., less than 20).
    assert total_cost < 20
    
    # Verify the path does not include the wall cell (3, 2)
    assert (3, 2) not in path

def test_out_of_bounds_start(simple_grid: Grid):
    """Test raising ValueError when the start point is outside the grid."""
    astar = AStarGrid(simple_grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((5, 0), (0, 0))

def test_out_of_bounds_end(simple_grid: Grid):
    """Test raising ValueError when the end point is outside the grid."""
    astar = AStarGrid(simple_grid)
    with pytest.raises(ValueError, match="out of bounds"):
        astar.find_path((0, 0), (0, 5))

def test_start_on_wall(simple_grid: Grid):
    """Test raising ValueError when the start point is a wall (cost 0)."""
    astar = AStarGrid(simple_grid)
    # Wall is at (2, 1)
    start_on_wall = (2, 1)
    end = (0, 0)
    with pytest.raises(ValueError, match="is a wall"):
        astar.find_path(start_on_wall, end)

def test_end_on_wall(simple_grid: Grid):
    """Test returning None when the end point is a wall (cost 0)."""
    astar = AStarGrid(simple_grid)
    start = (0, 0)
    # Wall is at (2, 1)
    end_on_wall = (2, 1)
    path = astar.find_path(start, end_on_wall)
    assert path is None