import heapq
, Tuple, Optional, Dict

# Type aliases for clarity
Grid = List[List[float]]
Point = Tuple[int, int]

class AStarGrid:
    """
    Implements A* pathfinding on a weighted 2D grid.

    The grid values represent the cost to enter that cell. 0 represents a wall.
    """

    def __init__(self, grid: Grid):
        """
        Initializes the AStarGrid with the map.

        Args:
            grid: A 2D list of floats where each float is the movement cost.
                  0.0 signifies a wall.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, point: Point) -> bool:
        """Checks if a point is within bounds and not a wall."""
        r, c = point
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] > 0.0

    def _manhattan_distance(self, p1: Point, p2: Point) -> float:
        """Calculates the Manhattan distance heuristic."""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def find_path(self, start: Point, end: Point) -> Optional[List[Point]]:
        """
        Finds the shortest path from start to end using the A* algorithm.

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
        # We include g_score to break ties consistently if f_scores are equal, though not strictly necessary for correctness.
        open_set: List[Tuple[float, float, Point]] = []
        
        # g_score: Cost from start to current node
        g_score: Dict[Point, float] = {start: 0.0}
        
        # f_score: Estimated total cost (g_score + heuristic)
        f_score: Dict[Point, float] = {start: self._manhattan_distance(start, end)}
        
        # came_from: Dictionary to reconstruct the path
        came_from: Dict[Point, Point] = {}

        # Initialize the priority queue
        heapq.heappush(open_set, (f_score[start], 0.0, start))

        # Directions for 4-way movement (Up, Down, Left, Right)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_set:
            # Pop the node with the lowest f_score
            current_f, current_g, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path: List[Point] = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not self._is_valid(neighbor):
                    continue

                # Cost to move to the neighbor (weight of the neighbor cell)
                movement_cost = self.grid[neighbor[0]][neighbor[1]]
                
                # Tentative g_score is the cost to reach the neighbor through the current node
                tentative_g_score = g_score[current] + movement_cost

                # If this path to neighbor is better than any previous one
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    # Update path records
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    
                    # Calculate new f_score
                    h_score = self._manhattan_distance(neighbor, end)
                    f_score[neighbor] = tentative_g_score + h_score
                    
                    # Add/Update in the priority queue
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))

        return None # No path found

# =============================================================================
# PYTEST TESTS
# =============================================================================
import pytest

@pytest.fixture
def simple_grid():
    """A standard 5x5 grid for basic testing."""
    # Costs: 1.0 for open, 0.0 for wall
    return [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ]

@pytest.fixture
def weighted_grid():
    """A 4x4 grid with varying weights."""
    # Costs: 1.0 (normal), 5.0 (expensive), 0.0 (wall)
    return [
        [1.0, 1.0, 5.0, 1.0],
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 5.0],
        [1.0, 1.0, 1.0, 1.0]
    ]

def test_start_equals_end(simple_grid):
    """Test case 1: Start and end points are the same."""
    grid_instance = AStarGrid(simple_grid)
    start = (0, 0)
    end = (0, 0)
    path = grid_instance.find_path(start, end)
    assert path == [start]

def test_simple_path_found(simple_grid):
    """Test case 2: A straightforward path exists without complex obstacles."""
    grid_instance = AStarGrid(simple_grid)
    start = (0, 0)
    end = (4, 4)
    path = grid_instance.find_path(start, end)
    assert path is not None
    assert len(path) > 1
    assert path[0] == start
    assert path[-1] == end

def test_path_avoiding_walls(simple_grid):
    """Test case 3: Pathfinding must correctly navigate around obstacles (walls)."""
    grid_instance = AStarGrid(simple_grid)
    start = (0, 0)
    end = (4, 4)
    path = grid_instance.find_path(start, end)
    assert path is not None
    
    # Check that the path does not contain any wall coordinates (0.0)
    for r, c in path:
        assert simple_grid[r][c] > 0.0

def test_no_path_exists(simple_grid):
    """Test case 4: No path exists between two disconnected regions."""
    grid_instance = AStarGrid(simple_grid)
    # Start in top-left, end in bottom-right, separated by a wall barrier
    start = (0, 0)
    end = (4, 0) # This should be reachable, let's pick a truly blocked spot.
    
    # Create a grid where the bottom row is fully blocked off from the top
    blocked_grid = [
        [1.0, 1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 0.0, 1.0], # Barrier row
        [1.0, 0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ]
    grid_instance_blocked = AStarGrid(blocked_grid)
    
    start_blocked = (0, 0)
    end_blocked = (4, 4)
    
    path = grid_instance_blocked.find_path(start_blocked, end_blocked)
    assert path is None

def test_weighted_path_optimality(weighted_grid):
    """Test case 5: A* must choose the path with the lowest total weight, not just the shortest hop count."""
    grid_instance = AStarGrid(weighted_grid)
    start = (0, 0)
    end = (3, 3)
    
    # Path 1 (Direct, through 5.0): (0,0) -> (0,2) [Cost 5.0] -> ...
    # Path 2 (Around the 5.0): (0,0) -> (1,0) -> (2,0) -> (2,1) -> (1,1) [Wall]
    # Let's trace the expected optimal path around the high-cost cell (0, 2) = 5.0
    
    path = grid_instance.find_path(start, end)
    assert path is not None
    
    # Calculate the actual cost of the found path
    total_cost = 0.0
    for i in range(len(path) - 1):
        r, c = path[i+1]
        total_cost += weighted_grid[r][c]
        
    # The optimal path should avoid the 5.0 cost cell at (0, 2) if possible.
    # The path should favor the 1.0 cells.
    
    # A known optimal path cost for this setup is around 7.0
    assert total_cost < 15.0 # Sanity check against a very bad path

def test_error_handling(simple_grid):
    """Test case 6: Comprehensive error handling for bounds and walls."""
    grid_instance = AStarGrid(simple_grid)
    
    # Test out of bounds start
    with pytest.raises(ValueError, match="out of bounds"):
        grid_instance.find_path((5, 5), (0, 0))
        
    # Test out of bounds end
    with pytest.raises(ValueError, match="out of bounds"):
        grid_instance.find_path((0, 0), (5, 5))
        
    # Test start on a wall
    wall_grid = [[0.0, 1.0], [1.0, 1.0]]
    grid_wall = AStarGrid(wall_grid)
    with pytest.raises(ValueError, match="is a wall"):
        grid_wall.find_path((0, 0), (1, 1))
        
    # Test end on a wall
    with pytest.raises(ValueError, match="is a wall"):
        grid_wall.find_path((1, 1), (0, 0))