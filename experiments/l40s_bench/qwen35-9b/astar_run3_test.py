import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.
    
    Movement is restricted to 4 directions (up, down, left, right).
    Grid values: 0 = impassable wall, positive int = cost to enter.
    """
    
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the grid.
        
        Args:
            grid: 2D list where grid[r][c] is the cost to enter cell (r, c).
                  0 represents a wall.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """
        Get valid 4-directional neighbors.
        
        Returns:
            List of (row, col) tuples for valid neighbors.
        """
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self._is_valid(nr, nc):
                # 0 is a wall, so we cannot move into it
                if self.grid[nr][nc] != 0:
                    neighbors.append((nr, nc))
        return neighbors

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance heuristic.
        
        Args:
            a: Current coordinates (row, col).
            b: Target coordinates (row, col).
            
        Returns:
            Manhattan distance: |a[0] - b[0]| + |a[1] - b[1]|
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A*.
        
        Args:
            start: Starting coordinates (row, col).
            end: Target coordinates (row, col).
            
        Returns:
            List of coordinates representing the path from start to end inclusive,
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate bounds
        if not self._is_valid(start[0], start[1]):
            raise ValueError(f"Start coordinates {start} are out of bounds.")
        if not self._is_valid(end[0], end[1]):
            raise ValueError(f"End coordinates {end} are out of bounds.")
        
        # Check for walls at start or end
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None

        # Edge case: Start equals End
        if start == end:
            return [start]

        # Priority Queue: (f_score, counter, current_node)
        # Counter is used to break ties in f_score to ensure consistent ordering
        counter = 0
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, end), counter, start))
        
        # To reconstruct path: map node -> (parent_node, cost_to_enter_node)
        came_from = {}
        g_score = {start: 0}
        
        while open_set:
            _, _, current = heapq.heappop(open_set)
            current_r, current_c = current

            # If we reached the end
            if current == end:
                # Reconstruct path
                path = []
                while current in came_from or current == start:
                    path.append(current)
                    if current == start:
                        break
                    current = came_from[current][0]
                return path[::-1]

            # Explore neighbors
            for neighbor in self._get_neighbors(current_r, current_c):
                neighbor_r, neighbor_c = neighbor
                tentative_g = g_score[current] + self.grid[neighbor_r][neighbor_c]

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = (current, self.grid[neighbor_r][neighbor_c])
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    counter += 1
                    heapq.heappush(open_set, (f_score, counter, neighbor))

        return None

import pytest
from typing import List, Tuple, Optional

def calculate_path_cost(path: List[Tuple[int, int]], grid: List[List[int]]) -> int:
    """Helper to calculate total cost of a path."""
    if not path:
        return 0
    total_cost = 0
    # Cost to enter the first cell is included in the grid logic, 
    # but strictly speaking, A* sums the cost of entering each cell from start to end.
    # In this implementation, g_score[start] is 0, and we add grid[neighbor] when moving.
    # So total cost = sum of costs of all cells in path except the start cell.
    for r, c in path[1:]:
        total_cost += grid[r][c]
    return total_cost

class TestAStarGrid:
    
    def test_simple_path_uniform_grid(self):
        """Test 1: Simple path on a uniform grid."""
        grid = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 3)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == end
        assert len(path) == 4  # (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(2,3) is 6 steps? 
        # Wait, shortest is (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3) (5 steps) 
        # or (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(2,3) (5 steps).
        # Manhattan dist is |2-0| + |3-0| = 5. Path length should be 6 (nodes).
        assert len(path) == 6
        
        # Verify optimality (Manhattan distance * cost 1 = 5)
        cost = calculate_path_cost(path, grid)
        assert cost == 5

    def test_path_around_obstacles(self):
        """Test 2: Path around obstacles."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (0, 3)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        # Must go down, right, right, up (or similar)
        # (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2) -> (2,3) -> (1,3) -> (0,3)
        # Length: 8 nodes. Cost: 7 (all 1s).
        assert len(path) == 8
        cost = calculate_path_cost(path, grid)
        assert cost == 7

    def test_weighted_grid_optimality(self):
        """Test 3: Weighted grid where path prefers lower-cost cells."""
        # Layout:
        # 1  10  1
        # 1  100 1
        # 1  1   1
        # Start (0,0), End (2,2).
        # Direct right path: (0,0)->(0,1)[10]->(0,2)[1] = Cost 11.
        # Down path: (0,0)->(1,0)[1]->(2,0)[1]->(2,1)[1]->(2,2)[1] = Cost 4.
        grid = [
            [1, 10, 1],
            [1, 100, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 2)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        # Expected path: (0,0) -> (1,0) -> (2,0) -> (2,1) -> (2,2)
        expected_path = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
        assert path == expected_path
        
        # Verify cost is minimal (4)
        cost = calculate_path_cost(path, grid)
        assert cost == 4

    def test_no_path_exists(self):
        """Test 4: No path exists (fully blocked)."""
        grid = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (0, 2)
        
        path = astar.find_path(start, end)
        assert path is None

    def test_start_equals_end(self):
        """Test 5: Start equals end."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        start = (1, 1)
        end = (1, 1)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        assert path == [start]

    def test_invalid_coordinates(self):
        """Test 6: Invalid coordinates raise ValueError."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        start = (5, 5)
        end = (0, 0)
        
        with pytest.raises(ValueError):
            astar.find_path(start, end)
            
        start = (0, 0)
        end = (5, 5)
        
        with pytest.raises(ValueError):
            astar.find_path(start, end)
            
        # Test start is wall
        grid_wall = [[0, 1], [1, 1]]
        astar_wall = AStarGrid(grid_wall)
        with pytest.raises(ValueError):
            astar_wall.find_path((0, 0), (1, 1))
            
        # Test end is wall
        with pytest.raises(ValueError):
            astar_wall.find_path((1, 1), (0, 0))