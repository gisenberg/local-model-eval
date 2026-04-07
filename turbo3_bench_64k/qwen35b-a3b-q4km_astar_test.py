import heapq
from typing import List, Tuple, Optional, Dict, Set

class AStarGrid:
    """
    A class to perform A* pathfinding on a 2D weighted grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initializes the AStarGrid with a 2D grid.

        Args:
            grid: A 2D list of integers where 0 is impassable and >0 is the movement cost.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Finds the shortest path from start to end using the A* algorithm.

        Args:
            start: (row, col) tuple representing the starting position.
            end: (row, col) tuple representing the destination position.

        Returns:
            A list of (row, col) tuples representing the path, or None if no path exists.

        Raises:
            ValueError: If start or end coordinates are out of the grid boundaries.
        """
        # 1. Bounds Check
        for r, c in [start, end]:
            if not (0 <= r < self.rows and 0 <= c < self.cols):
                raise ValueError("Start or end coordinates are out of bounds.")

        # 2. Edge Case: Start or End is a wall
        if self.grid[start[0]][start[1]] == 0 or self.sgrid_is_wall(end):
            return None

        # 3. Edge Case: Start equals End
        if start == end:
            return [start]

        # 4. A* Initialization
        # open_set stores: (f_score, current_node)
        open_set: List[Tuple[float, Tuple[int, int]]] = []
        heapq.heappush(open_set, (self._heuristic(start, end), start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score[node] is the cost of the cheapest path from start to node currently known.
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        
        # Track visited nodes to avoid redundant processing
        visited: Set[Tuple[int, int]] = set()

        while open_set:
            _, current = heapq.heappop(open_open_set_node(open_set))
            
            if current == end:
                return self._reconstruct_path(came_from, current)

            if current in visited:
                continue
            visited.add(current)

            for neighbor in self._get_neighbors(current):
                # Cost to enter the neighbor cell
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g_score = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, neighbor))

        return None

    def _sgrid_is_wall(self, pos: Tuple[int, int]) -> bool:
        """Checks if a cell is a wall."""
        return self.grid[pos[0]][pos[1]] == 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Returns valid 4-directional neighbors."""
        r, c = pos
        neighbors = []
        # Up, Down, Left, Right
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] > 0:  # Not a wall
                    neighbors.append((nr, nc))
        return neighbors

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], 
                         current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Backtracks from end to start to build the path list."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def _open_open_set_node(self, open_set: List) -> Tuple[float, Tuple[int, int]]:
        """Helper to pop from heap safely."""
        return heapq.heappop(open_set)

# --- Pytest Suite ---
import pytest

def test_simple_path():
    """Test a simple path on a uniform grid with no obstacles."""
    grid = [[1, 1, 1], 
            [1, 1, 1], 
            [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Shortest path in 4-dir on uniform grid is 4 steps (5 nodes total)
    assert len(path) == 5
    # Total cost: sum of weights of nodes entered (excluding start)
    # (0,0)->(0,1)->(0,2)->(1,2)->(2,2) => 1+1+1+1 = 4
    cost = sum(grid[r][c] for r, c in path[1:])
    assert cost == 4

def test_path_around_obstacles():
    """Test that the algorithm navigates around walls."""
    grid = [[1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    # Must go around the 0 at (1,1)
    assert (1, 1) not in path

def test_weighted_grid_optimality():
    """Test that the algorithm prefers a longer path with lower total cost."""
    # Direct path (0,0)->(0,1)->(0,2) costs 10 + 10 = 20
    # Longer path (0,0)->(1,0)->(1,1)->(1,2)->(0,2) costs 1+1+1+1 = 4
    grid = [[1, 10, 1],
            [1, 1, 1],
            [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is not None
    cost = sum(grid[r][c] for r, c in path[1:])
    assert cost == 4 

def test_no_path_exists():
    """Test scenario where end is completely blocked."""
    grid = [[1, 1, 1],
            [1, 0, 0],
            [1, 0, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is None

def test_start_equals_end():
    """Test scenario where start and end are the same."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_invalid_coordinates():
    """Test that out of bounds coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (5, 5))

if __name__ == "__main__":
    # This allows running the script directly to see if tests pass
    pytest.main([__file__])