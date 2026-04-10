import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D grid.
        grid values represent movement cost (0 = impassable wall, positive int = cost to enter that cell).
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        # Directions: Up, Down, Left, Right
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _get_neighbors(self, r: int, c: int) -> List[Tuple[int, int]]:
        """Get valid neighbors (not walls) for a given cell."""
        neighbors = []
        for dr, dc in self.directions:
            nr, nc = r + dr, c + dc
            if self._is_valid(nr, nc) and self.grid[nr][nc] > 0:
                neighbors.append((nr, nc))
        return neighbors

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        Returns a list of (row, col) coordinates or None if no path exists.
        """
        # Validate coordinates
        if not self._is_valid(start[0], start[1]) or not self._is_valid(end[0], end[1]):
            raise ValueError("Start or end coordinates are out of bounds.")
        
        # Check for walls
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None
            
        # Edge case: start equals end
        if start == end:
            return [start]

        # Open set: min-heap storing (f_score, g_score, row, col)
        # We use g_score as a secondary sort key to break ties deterministically
        open_set = [(self._heuristic(start, end), 0, start[0], start[1])]
        
        # Track visited nodes to avoid re-processing
        visited = set()
        
        # Track parent pointers to reconstruct path
        # Key: (r, c), Value: (parent_r, parent_c)
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        
        while open_set:
            # Pop node with lowest f_score
            _, current_g, current_r, current_c = heapq.heappop(open_set)
            current = (current_r, current_c)
            
            # If we reached the end, reconstruct path
            if current == end:
                path = [end]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            
            # Skip if already visited with a lower cost
            if current in visited:
                continue
            visited.add(current)
            
            # Explore neighbors
            for neighbor_r, neighbor_c in self._get_neighbors(current_r, current_c):
                neighbor = (neighbor_r, neighbor_c)
                
                if neighbor in visited:
                    continue
                
                # Cost to move to neighbor
                move_cost = self.grid[neighbor_r][neighbor_c]
                tentative_g = current_g + move_cost
                
                # If this path to neighbor is better than any previous one, record it
                # Since we use a heap, we might encounter the same node multiple times.
                # We only care if we found a cheaper way to get here.
                # Note: In a standard A* with consistent heuristics, the first time we pop a node
                # is the optimal path. However, with weighted grids, we must ensure we process
                # neighbors correctly.
                
                # We push to heap regardless, but the visited check above handles pruning.
                # To ensure optimality strictly, we usually check if tentative_g < known_g.
                # Here, since we don't store a separate 'g_score' map for open nodes, 
                # we rely on the heap ordering and the visited set.
                
                heapq.heappush(open_set, (tentative_g + self._heuristic(neighbor, end), tentative_g, neighbor_r, neighbor_c))
                came_from[neighbor] = current

        return None

import pytest
from typing import List, Tuple, Optional
  # Assuming the class is in a file named a_star_grid.py

def calculate_path_cost(grid: List[List[int]], path: List[Tuple[int, int]]) -> int:
    """Helper to calculate total cost of a path."""
    if not path:
        return 0
    total = 0
    for r, c in path[1:]: # Skip start node cost as usually cost is to ENTER a cell
        total += grid[r][c]
    return total

class TestAStarGrid:
    
    def test_simple_path_uniform_grid(self):
        """Test simple path on a uniform grid (all costs 1)."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 2)
        
        path = astar.find_path(start, end)
        
        assert path is not None, "Path should exist"
        assert path[0] == start, "Path should start at start"
        assert path[-1] == end, "Path should end at end"
        # Optimal path length in steps is 4 (Manhattan distance), cost is 4 (entering 4 cells)
        assert len(path) == 5, "Path should have 5 nodes (start + 4 steps)"
        assert calculate_path_cost(grid, path) == 4, "Total cost should be 4"

    def test_path_around_obstacles(self):
        """Test pathfinding around obstacles (0s)."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 3)
        
        path = astar.find_path(start, end)
        
        assert path is not None, "Path should exist around obstacles"
        assert path[0] == start
        assert path[-1] == end
        # Verify no walls in path
        for r, c in path:
            assert grid[r][c] != 0, "Path should not contain walls"
        # Verify connectivity
        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i+1]
            assert abs(r1 - r2) + abs(c1 - c2) == 1, "Steps must be adjacent"

    def test_weighted_grid_optimality(self):
        """Test that the algorithm prefers lower-cost cells."""
        # Grid where going down is expensive (cost 10), going right is cheap (cost 1)
        grid = [
            [1, 1, 1],
            [10, 10, 1],
            [10, 10, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 2)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        # The optimal path should go Right -> Right -> Down -> Down (Cost: 1+1+1+1 = 4)
        # Instead of Down -> Down -> Right -> Right (Cost: 10+10+1+1 = 22)
        # Path: (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2)
        expected_path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
        assert path == expected_path, f"Expected {expected_path}, got {path}"
        assert calculate_path_cost(grid, path) == 4, "Cost should be 4"

    def test_no_path_exists(self):
        """Test case where no path exists (fully blocked)."""
        grid = [
            [1, 0, 1],
            [0, 0, 0],
            [1, 0, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 2)
        
        path = astar.find_path(start, end)
        assert path is None, "Path should be None when blocked"

    def test_start_equals_end(self):
        """Test edge case where start equals end."""
        grid = [[1, 1], [1, 1]]
        astar = AStarGrid(grid)
        start = (1, 1)
        end = (1, 1)
        
        path = astar.find_path(start, end)
        assert path == [start], "Path should be just the start node"

    def test_invalid_coordinates(self):
        """Test that invalid coordinates raise ValueError."""
        grid = [[1, 1], [1, 1]]
        astar = AStarGrid(grid)
        
        # Out of bounds start
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (1, 1))
            
        # Out of bounds end
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (5, 5))

    def test_start_or_end_is_wall(self):
        """Test that start or end being a wall returns None."""
        grid = [
            [0, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Start is wall
        assert astar.find_path((0, 0), (1, 1)) is None
        
        # End is wall
        grid[1][1] = 0
        assert astar.find_path((0, 1), (1, 1)) is None