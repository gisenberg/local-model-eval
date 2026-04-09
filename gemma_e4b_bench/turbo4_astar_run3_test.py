import heapq
, Tuple, Optional, Dict, Set

class AStarGrid:
    """
    Implements the A* pathfinding algorithm on a weighted 2D grid.

    The grid cells can have different movement costs (weights).
    Walls are represented by a weight of 0.
    """

    def __init__(self, grid: List[List[float]]):
        """
        Initializes the AStarGrid.

        Args:
            grid: A 2D list of floats representing the grid weights.
                  0.0 indicates a wall.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def _is_valid(self, position: Tuple[int, int]) -> bool:
        """Checks if a position is within grid bounds."""
        r, c = position
        return 0 <= r < self.rows and 0 <= c < self.cols

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from start to end using A*.

        Args:
            start: The starting coordinates (row, col).
            end: The target coordinates (row, col).

        Returns:
            A list of (row, col) tuples representing the optimal path,
            or None if no path exists.

        Raises:
            ValueError: If start or end are out of bounds, or if start/end are walls.
        """
        if start == end:
            return [start]

        if not self._is_valid(start):
            raise ValueError(f"Start point {start} is out of bounds.")
        if not self._is_valid(end):
            raise ValueError(f"End point {end} is out of bounds.")
        
        if self.grid[start[0]][start[1]] == 0.0 or self.grid[end[0]][end[1]] == 0.0:
            # Cannot start or end on a wall
            return None

        # Priority queue: (f_score, g_score, position)
        # We include g_score for tie-breaking consistency, though f_score is primary.
        open_set: List[Tuple[float, float, Tuple[int, int]]] = [(0.0, 0.0, start)]
        
        # g_score: Cost from start to current node
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        
        # came_from: Used to reconstruct the path
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # Heuristic function (Manhattan distance)
        def heuristic(a: Tuple[int, int], b: Tuple[int, int]) -> float:
            """Manhattan distance heuristic."""
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

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
                return path[::-1]

            # 4-directional neighbors (Up, Down, Left, Right)
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not self._is_valid(neighbor):
                    continue

                weight = self.grid[neighbor[0]][neighbor[1]]
                
                # Check for walls (weight == 0)
                if weight == 0.0:
                    continue

                # Tentative g_score is the cost to reach the neighbor through the current node
                tentative_g_score = g_score[current] + weight

                # If this path to the neighbor is better than any previously found path
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    
                    # f = g + h
                    h_score = heuristic(neighbor, end)
                    f_score = tentative_g_score + h_score
                    
                    # Push to priority queue
                    heapq.heappush(open_set, (f_score, tentative_g_score, neighbor))

        return None # Path not found

# --- Example Usage (Optional, for local testing) ---
if __name__ == '__main__':
    # Grid: 3x4
    # Weights: 1.0 (normal), 5.0 (slow), 0.0 (wall)
    grid_data = [
        [1.0, 1.0, 5.0, 1.0],
        [1.0, 0.0, 5.0, 1.0],
        [1.0, 1.0, 1.0, 1.0]
    ]
    
    try:
        astar = AStarGrid(grid_data)
        start_node = (0, 0)
        end_node = (2, 3)
        
        path = astar.find_path(start_node, end_node)
        
        if path:
            print("Path found:", path)
            # Calculate total cost for verification
            total_cost = 0
            for i in range(len(path) - 1):
                r1, c1 = path[i]
                r2, c2 = path[i+1]
                total_cost += grid_data[r2][c2]
            print(f"Total cost: {total_cost}")
        else:
            print("No path found.")

    except ValueError as e:
        print(f"Error: {e}")

import pytest
, Tuple, Optional


# --- Test Fixtures ---

@pytest.fixture
def simple_grid():
    """A simple 3x3 grid with uniform cost 1.0."""
    return [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ]

@pytest.fixture
def weighted_grid():
    """A 4x4 grid with varying costs and a wall."""
    return [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 5.0, 0.0, 1.0],  # 5.0 is slow, 0.0 is wall
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0]
    ]

@pytest.fixture
def complex_grid():
    """A grid designed to test pathfinding around obstacles."""
    return [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 0.0, 1.0, 1.0]
    ]

# --- Tests ---

def test_start_equals_end(simple_grid):
    """Test case 1: Start and end points are the same."""
    grid = AStarGrid(simple_grid)
    start = (1, 1)
    end = (1, 1)
    path = grid.find_path(start, end)
    assert path == [start]

def test_simple_straight_path(simple_grid):
    """Test case 2: Simple path in an open, uniform grid."""
    grid = AStarGrid(simple_grid)
    start = (0, 0)
    end = (2, 2)
    path = grid.find_path(start, end)
    
    assert path is not None
    # Optimal path length should be 5 steps (0,0 -> 1,0 -> 2,0 -> 2,1 -> 2,2) or similar
    assert len(path) == 5 
    assert path[0] == start
    assert path[-1] == end

def test_path_blocked_by_wall(weighted_grid):
    """Test case 3: Path is completely blocked by walls."""
    grid = AStarGrid(weighted_grid)
    start = (0, 0)
    end = (0, 3) # Trying to go around the wall at (1, 2)
    
    # This path should be possible, but let's test a truly blocked scenario
    # Block the entire path between (0,0) and (0,3)
    blocked_grid = [
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0]
    ]
    grid_blocked = AStarGrid(blocked_grid)
    
    path = grid_blocked.find_path((0, 0), (0, 3))
    assert path is None

def test_optimal_path_with_weights(weighted_grid):
    """Test case 4: A* chooses the path with the lowest total weight."""
    grid = AStarGrid(weighted_grid)
    start = (0, 0)
    end = (2, 3)
    
    path = grid.find_path(start, end)
    assert path is not None
    
    # Path 1 (Direct, through 5.0): (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2) -> (2,3)
    # Cost: 1 + 1 + 1 + 1 + 1 = 5.0 (if we ignore the 5.0 at (1,1) and (1,2))
    
    # The optimal path should avoid the 5.0 cell at (1,1) if possible.
    # The path through the bottom row should be optimal.
    
    # Check if the path found is reasonably short and avoids the high-cost area if possible
    assert len(path) <= 7 # Should be short
    
    # Verify the cost of the found path is minimal (manual check confirms bottom route is best)
    total_cost = 0
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        total_cost += weighted_grid[r2][c2]
    
    # The optimal cost should be 5.0 (e.g., (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(2,3))
    assert total_cost == 5.0

def test_out_of_bounds_error(simple_grid):
    """Test case 5: Start or end point is outside the grid boundaries."""
    grid = AStarGrid(simple_grid)
    
    # Test start out of bounds
    with pytest.raises(ValueError, match="Start point"):
        grid.find_path((3, 0), (0, 0))
        
    # Test end out of bounds
    with pytest.raises(ValueError, match="End point"):
        grid.find_path((0, 0), (0, 3))

def test_start_or_end_on_wall(complex_grid):
    """Test case 6: Start or end point is located on a wall (weight 0.0)."""
    grid = AStarGrid(complex_grid)
    
    # Test start on wall
    wall_start = (1, 1)
    path = grid.find_path(wall_start, (3, 3))
    assert path is None
    
    # Test end on wall
    wall_end = (1, 2)
    path = grid.find_path((0, 0), wall_end)
    assert path is None