import heapq
from typing import List, Tuple, Optional, Dict

class AStarGrid:
    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D grid of movement costs.
        
        Args:
            grid: A 2D list where 0 represents a wall (impassable) 
                  and positive integers represent the cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0
        self._validate_grid_dimensions()

    def _validate_grid_dimensions(self):
        """Ensure all rows have the same length."""
        if self.rows == 0:
            return
        first_len = self.cols
        for r in range(self.rows):
            if len(self.grid[r]) != first_len:
                raise ValueError("Grid rows must have equal length.")

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if a coordinate is within bounds and not a wall."""
        r, c = pos
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        # 0 represents a wall
        if self.grid[r][c] == 0:
            return False
        return True

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid 4-directional neighbors (up, down, left, right).
        """
        r, c = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if self._is_valid((nr, nc)):
                neighbors.append((nr, nc))
        return neighbors

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two points.
        Used as the heuristic function h(n).
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Tuple (row, col) representing the starting position.
            end: Tuple (row, col) representing the target position.
            
        Returns:
            A list of (row, col) tuples representing the path from start to end inclusive,
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # 1. Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start position {start} is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End position {end} is out of bounds.")

        # 2. Handle edge cases: Start or End is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # 3. Handle edge case: Start equals End
        if start == end:
            return [start]

        # Priority Queue: (f_score, g_score, current_pos)
        # We use g_score as a tie-breaker in the tuple to ensure stability, 
        # though strictly f_score is the primary sort key.
        open_set: List[Tuple[int, int, Tuple[int, int]]] = []
        heapq.heappush(open_set, (0, 0, start))

        # Track the best g_score found so far for each node
        g_scores: Dict[Tuple[int, int], int] = {start: 0}
        
        # Track parent pointers to reconstruct the path
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # Closed set (visited nodes)
        closed_set: set = set()

        while open_set:
            # Pop the node with the lowest f_score
            _, current_g, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            
            if current == end:
                # Reconstruct path
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]

            closed_set.add(current)

            for neighbor in self._get_neighbors(current):
                if neighbor in closed_set:
                    continue

                # Cost to move to neighbor (cost of the neighbor cell)
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = current_g + move_cost

                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor))
                    came_from[neighbor] = current

        return None

import pytest


class TestAStarGrid:
    def test_simple_path_uniform_grid(self):
        """Test a simple path on a uniform cost grid (all 1s)."""
        grid = [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 2)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == end
        # Manhattan distance is 4 steps, cost = 4 * 1 = 4 (excluding start cost usually, 
        # but here cost is to ENTER. Path length 5 nodes -> 4 moves. 
        # Cost = sum of costs of nodes 1 to 4.
        # Path: (0,0)->(0,1)->(0,2)->(0,3)->(1,3)->(2,3)->(2,2) is one option.
        # Shortest is 4 steps. Total cost = 4 * 1 = 4.
        total_cost = sum(grid[r][c] for r, c in path[1:])
        assert total_cost == 4

    def test_path_around_obstacles(self):
        """Test pathfinding around walls (0s)."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (2, 3)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        assert path[0] == start
        assert path[-1] == end
        
        # Verify no walls in path
        for r, c in path:
            assert grid[r][c] != 0

    def test_weighted_grid_optimality(self):
        """Test that the algorithm prefers lower-cost cells over shorter distance."""
        # Grid where going straight is expensive, but going around is cheap
        grid = [
            [1, 10, 10, 1],
            [1, 1, 1, 1],
            [1, 10, 10, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (0, 3)
        
        path = astar.find_path(start, end)
        
        assert path is not None
        
        # Calculate cost of the found path
        total_cost = sum(grid[r][c] for r, c in path[1:])
        
        # Direct path (0,0)->(0,1)->(0,2)->(0,3) cost: 10 + 10 + 1 = 21
        # Bottom path (0,0)->(1,0)->(1,1)->(1,2)->(1,3)->(0,3) cost: 1+1+1+1+1 = 5
        # The algorithm should choose the bottom path.
        assert total_cost == 5

    def test_no_path_exists(self):
        """Test when the destination is fully blocked."""
        grid = [
            [1, 0, 1],
            [1, 0, 1],
            [1, 0, 1]
        ]
        astar = AStarGrid(grid)
        start = (0, 0)
        end = (0, 2)
        
        path = astar.find_path(start, end)
        assert path is None

    def test_start_equals_end(self):
        """Test when start and end coordinates are the same."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        start = (1, 1)
        end = (1, 1)
        
        path = astar.find_path(start, end)
        assert path == [start]

    def test_invalid_coordinates(self):
        """Test that ValueError is raised for out-of-bounds coordinates."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Test start out of bounds
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (1, 1))
            
        # Test end out of bounds
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (5, 5))
            
        # Test start on a wall (should return None, not raise)
        grid_wall = [[0, 1], [1, 1]]
        astar_wall = AStarGrid(grid_wall)
        path = astar_wall.find_path((0, 0), (1, 1))
        assert path is None