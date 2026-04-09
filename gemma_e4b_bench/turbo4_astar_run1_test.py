import heapq
, Tuple, Optional, Dict

# Define types for clarity
Point = Tuple[int, int]
Grid = List[List[float]]

class AStarGrid:
    """
    Implements the A* pathfinding algorithm on a weighted 2D grid.

    The grid cells can have different weights, representing the cost to traverse them.
    Walls are represented by a weight of 0.
    """

    def __init__(self, grid: Grid):
        """
        Initializes the AStarGrid.

        Args:
            grid: A 2D list (list of lists) where each element is the traversal cost.
                  0 indicates a wall.
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
        x, y = current
        # 4-directional movement: (dx, dy)
        neighbors = [
            (x + 1, y),  # Right
            (x - 1, y),  # Left
            (x, y + 1),  # Down
            (x, y - 1)   # Up
        ]
        return neighbors

    def find_path(self, start: Point, end: Point) -> Optional[List[Point]]:
        """
        Finds the shortest path from the start point to the end point using A*.

        Args:
            start: The starting coordinates (row, col).
            end: The target coordinates (row, col).

        Returns:
            A list of coordinates representing the optimal path, or None if no path exists.

        Raises:
            ValueError: If start or end points are out of bounds or on a wall.
        """
        # 1. Handle start == end
        if start == end:
            return [start]

        # 2. Boundary and Wall Checks
        def is_valid(p: Point) -> bool:
            r, c = p
            return 0 <= r < self.rows and 0 <= c < self.cols

        if not is_valid(start) or self.grid[start[0]][start[1]] == 0:
            raise ValueError(f"Start point {start} is invalid or a wall.")
        if not is_valid(end) or self.grid[end[0]][end[1]] == 0:
            raise ValueError(f"End point {end} is invalid or a wall.")

        # Priority queue: stores (f_score, g_score, point)
        # We include g_score to break ties consistently, though f_score is primary.
        open_set: List[Tuple[float, float, Point]] = [(0.0, 0.0, start)]

        # g_score: Cost from start to current node
        g_score: Dict[Point, float] = {start: 0.0}
        
        # f_score: Estimated total cost (g_score + heuristic)
        f_score: Dict[Point, float] = {start: self._heuristic(start, end)}
        
        # came_from: Used to reconstruct the path
        came_from: Dict[Point, Point] = {}

        while open_set:
            # Pop the node with the lowest f_score
            current_f, current_g, current = heapq.heappop(open_set)

            if current == end:
                # Path found, reconstruct and return
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]  # Reverse to get start -> end order

            for neighbor in self._get_neighbors(current):
                if not is_valid(neighbor) or self.grid[neighbor[0]][neighbor[1]] == 0:
                    continue  # Skip out-of-bounds or walls

                # Cost to move to the neighbor (weight of the neighbor cell)
                movement_cost = self.grid[neighbor[0]][neighbor[1]]
                
                # Tentative g_score is the cost to reach the neighbor through the current node
                tentative_g_score = g_score[current] + movement_cost

                # If this path to the neighbor is better than any previously found path
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    
                    # This path is better, record it
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    
                    h = self._heuristic(neighbor, end)
                    f_score[neighbor] = tentative_g_score + h
                    
                    # Push to priority queue. We push the new, better score.
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g_score, neighbor))

        # If the loop finishes without reaching the end
        return None

# --- Example Usage (Optional, for local testing) ---
if __name__ == '__main__':
    # Grid: 3x4
    # Weights: 1 (normal), 5 (expensive), 0 (wall)
    # (0,0) (0,1) (0,2) (0,3)
    # (1,0) (1,1) (1,2) (1,3)
    # (2,0) (2,1) (2,2) (2,3)
    grid_data = [
        [1, 1, 5, 1],
        [1, 0, 1, 1],  # (1,1) is a wall
        [1, 1, 1, 1]
    ]
    
    solver = AStarGrid(grid_data)
    
    start_point = (0, 0)
    end_point = (2, 3)
    
    path = solver.find_path(start_point, end_point)
    
    if path:
        print("Path found:", path)
        # Calculate total cost of the found path
        total_cost = 0
        for i in range(len(path)):
            r, c = path[i]
            # Cost is the weight of the cell itself (except start, which is handled by the first step)
            if i > 0:
                total_cost += grid_data[r][c]
        print(f"Total path cost (excluding start cell cost): {total_cost}")
    else:
        print("No path found.")

import pytest
, Tuple
, Point, Grid

# --- Fixtures for reusable test setups ---

@pytest.fixture
def simple_grid() -> Grid:
    """A simple 3x3 grid with uniform cost 1."""
    return [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]

@pytest.fixture
def weighted_grid() -> Grid:
    """A 4x4 grid with varying weights and a wall."""
    # Weights: 1 (normal), 5 (expensive), 0 (wall)
    return [
        [1, 1, 5, 1],
        [1, 0, 1, 1],  # (1,1) is a wall
        [1, 1, 1, 1],
        [1, 5, 1, 1]
    ]

@pytest.fixture
def grid_with_obstacle() -> Grid:
    """A grid where the path is forced around an obstacle."""
    return [
        [1, 1, 1, 1],
        [1, 0, 0, 1],  # Obstacle block
        [1, 1, 1, 1]
    ]

# --- Test Cases ---

def test_start_equals_end(simple_grid: Grid):
    """Test case 1: Start and end points are the same."""
    solver = AStarGrid(simple_grid)
    start = (1, 1)
    end = (1, 1)
    path = solver.find_path(start, end)
    assert path == [start]

def test_simple_path_found(simple_grid: Grid):
    """Test case 2: A straightforward path exists in a uniform grid."""
    solver = AStarGrid(simple_grid)
    start = (0, 0)
    end = (2, 2)
    path = solver.find_path(start, end)
    
    assert path is not None
    assert len(path) > 1
    assert path[0] == start
    assert path[-1] == end

def test_path_avoiding_wall(grid_with_obstacle: Grid):
    """Test case 3: Path must navigate around a block of walls."""
    solver = AStarGrid(grid_with_obstacle)
    start = (0, 0)
    end = (2, 3)
    path = solver.find_path(start, end)
    
    assert path is not None
    assert path[0] == start
    assert path[-1] == end
    
    # Ensure the path does not pass through the wall coordinates (1,1) or (1,2)
    for r, c in path:
        assert (r, c) not in [(1, 1), (1, 2)]

def test_optimal_path_with_weights(weighted_grid: Grid):
    """Test case 4: A* must choose the path with the lowest total weight."""
    solver = AStarGrid(weighted_grid)
    start = (0, 0)
    end = (2, 3)
    path = solver.find_path(start, end)
    
    assert path is not None
    
    # Expected optimal path should avoid the high-cost cell (0, 2) = 5
    # Path through (1, 2) [cost 1] is better than going through (0, 2) [cost 5]
    
    # Check that the path found is not trivially short but cost-effective
    assert len(path) > 3 
    
    # A rough check: the path should not contain the expensive cell (0, 2) if a cheaper route exists
    for r, c in path:
        if (r, c) == (0, 2):
            pytest.fail("A* chose the expensive path through (0, 2)")

def test_no_path_exists(grid_with_obstacle: Grid):
    """Test case 5: No path exists because the end is completely walled off."""
    solver = AStarGrid(grid_with_obstacle)
    start = (0, 0)
    # Target (1, 3) is reachable, but let's block it off completely
    grid_with_obstacle[0][3] = 0
    grid_with_obstacle[2][3] = 0
    
    # Now, the end point (2, 3) is unreachable from (0, 0)
    solver_blocked = AStarGrid(grid_with_obstacle)
    path = solver_blocked.find_path(start, (2, 3))
    
    assert path is None

def test_out_of_bounds_error(simple_grid: Grid):
    """Test case 6: Should raise ValueError for out-of-bounds start point."""
    solver = AStarGrid(simple_grid)
    
    # Test start out of bounds
    with pytest.raises(ValueError, match="Start point"):
        solver.find_path((3, 0), (1, 1))
        
    # Test end out of bounds
    with pytest.raises(ValueError, match="End point"):
        solver.find_path((0, 0), (1, 3))

def test_start_on_wall_error(weighted_grid: Grid):
    """Test case 7 (Bonus): Should raise ValueError if the start point is a wall (weight 0)."""
    solver = AStarGrid(weighted_grid)
    wall_point = (1, 1)
    
    with pytest.raises(ValueError, match="Start point"):
        solver.find_path(wall_point, (2, 3))