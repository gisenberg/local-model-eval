import heapq
from typing import List, Tuple, Optional, Set

class AStarGrid:
    """
    A* pathfinding implementation for a weighted 2D grid.
    Supports 4-directional movement (up, down, left, right).
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.
        
        Args:
            grid: A 2D list where 0 represents a wall (impassable) and 
                  positive integers represent the cost to enter that cell.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # Validate grid contents
        for r in range(self.rows):
            for c in range(self.cols):
                if grid[r][c] < 0:
                    raise ValueError(f"Grid cost at ({r}, {c}) cannot be negative.")

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[Tuple[int, int], int]]:
        """
        Get valid neighbors and their movement costs.
        
        Args:
            pos: Current (row, col) position.
            
        Returns:
            List of tuples: ((next_row, next_col), cost_to_enter).
        """
        r, c = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
        neighbors = []
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            
            # Check bounds
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                # Check if not a wall (0 cost)
                if self.grid[nr][nc] > 0:
                    neighbors.append(((nr, nc), self.grid[nr][nc]))
                    
        return neighbors

    def _heuristic(self, pos: Tuple[int, int], end: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance heuristic.
        
        Args:
            pos: Current position.
            end: Target position.
            
        Returns:
            Manhattan distance.
        """
        return abs(pos[0] - end[0]) + abs(pos[1] - end[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A*.
        
        Args:
            start: Starting (row, col) tuple.
            end: Ending (row, col) tuple.
            
        Returns:
            List of (row, col) tuples representing the path, or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start coordinate {start} is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End coordinate {end} is out of bounds.")

        # Edge case: Start equals End
        if start == end:
            return [start]

        # Check if start or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Open set: (f_score, g_score, current_pos)
        # We use a counter to break ties for heap stability if f_scores are equal
        counter = 0
        open_set = [(0, 0, start)] # (f, g, pos)
        
        came_from: dict = {}
        g_score: dict = {start: 0}
        f_score: dict = {start: self._heuristic(start, end)}
        
        open_set_hash: Set[Tuple[int, int]] = {start}

        while open_set:
            # Pop node with lowest f_score
            _, _, current = heapq.heappop(open_set)
            
            if current not in open_set_hash:
                continue
            
            if current == end:
                # Reconstruct path
                path = []
                curr = end
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.append(start)
                return path[::-1]

            open_set_hash.remove(current)

            for neighbor, move_cost in self._get_neighbors(current):
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    f_score[neighbor] = f
                    
                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f, counter, neighbor))
                        open_set_hash.add(neighbor)

        return None

import pytest
from typing import List, Tuple, Optional


# Helper to calculate total cost of a path
def calculate_path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    if not path:
        return 0
    total = 0
    for r, c in path:
        total += grid[r][c]
    return total

class TestAStarGrid:
    
    @pytest.fixture
    def uniform_grid(self):
        """A simple 5x5 grid with no walls."""
        return [[1 for _ in range(5)] for _ in range(5)]

    @pytest.fixture
    def weighted_grid(self):
        """A grid with varying costs and walls."""
        return [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1]
        ]

    @pytest.fixture
    def low_cost_path_grid(self):
        """Grid designed to prefer a longer path with lower costs."""
        return [
            [1, 1, 1, 1, 1],
            [1, 10, 10, 10, 1],
            [1, 1, 1, 1, 1],
            [1, 10, 10, 10, 1],
            [1, 1, 1, 1, 1]
        ]

    def test_simple_path_uniform_grid(self, uniform_grid):
        """Test finding a simple path on a uniform grid."""
        astar = AStarGrid(uniform_grid)
        start = (0, 0)
        end = (2, 2)
        path = astar.find_path(start, end)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == end
        
        # Verify path validity (adjacent cells)
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            assert abs(r1 - r2) + abs(c1 - c2) == 1

    def test_path_around_obstacles(self, weighted_grid):
        """Test pathfinding around walls (0s)."""
        astar = AStarGrid(weighted_grid)
        start = (0, 0)
        end = (4, 4)
        path = astar.find_path(start, end)
        
        assert path is not None
        # Ensure no cell in path is a wall
        for r, c in path:
            assert weighted_grid[r][c] > 0, f"Path went through a wall at ({r}, {c})"

    def test_weighted_grid_optimality(self, low_cost_path_grid):
        """Test that the algorithm prefers lower-cost cells even if path is longer."""
        astar = AStarGrid(low_cost_path_grid)
        start = (0, 0)
        end = (4, 4)
        path = astar.find_path(start, end)
        
        assert path is not None
        
        # Calculate cost of found path
        found_cost = calculate_path_cost(low_cost_path_grid, path)
        
        # The optimal path should go around the high-cost center (cost 10)
        # Direct path through center would be much more expensive.
        # We assert that the cost is reasonable (e.g., < 20, whereas a center path would be > 30)
        assert found_cost < 20, f"Path is not optimal. Cost: {found_cost}"

    def test_no_path_exists(self):
        """Test when start and end are separated by walls."""
        grid = [
            [1, 1, 1],
            [0, 0, 0],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 2)
        
        path = astar.find_path(start, end)
        assert path is None

    def test_start_equals_end(self):
        """Test edge case where start and end are the same."""
        grid = [[1, 1], [1, 1]]
        astar = AStarGrid(grid)
        start = (0, 1)
        end = (0, 1)
        
        path = astar.find_path(start, end)
        assert path == [start]

    def test_invalid_coordinates(self):
        """Test that out of bounds coordinates raise ValueError."""
        grid = [[1, 1], [1, 1]]
        astar = AStarGrid(grid)
        
        # Start out of bounds
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (1, 1))
            
        # End out of bounds
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (5, 5))

    def test_start_or_end_is_wall(self):
        """Test that if start or end is a wall, path is None."""
        grid = [
            [1, 1],
            [1, 0]
        ]
        astar = AStarGrid(grid)
        
        # Start is wall
        assert astar.find_path((1, 1), (0, 0)) is None
        
        # End is wall
        assert astar.find_path((0, 0), (1, 1)) is None