import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding algorithm on a weighted 2D grid.
    
    Grid representation:
        - 0 represents a wall/impassable cell
        - Positive numbers represent traversal costs (weights)
    """
    
    def __init__(self, grid: List[List[float]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """Check if position is within grid boundaries."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, pos: Tuple[int, int]) -> bool:
        """Check if position is a wall (0)."""
        return self.grid[pos[0]][pos[1]] == 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Manhattan distance heuristic for 4-directional movement."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.
        
        Args:
            start: Tuple of (row, col) for starting position.
            end: Tuple of (row, col) for target position.
            
        Returns:
            List of (row, col) tuples representing the optimal path, 
            or None if no valid path exists.
            
        Raises:
            ValueError: If start or end is out of bounds or on a wall.
        """
        if not self._is_valid(start) or not self._is_valid(end):
            raise ValueError("Start or end position is out of bounds.")
        if self._is_wall(start) or self._is_wall(end):
            raise ValueError("Start or end position is a wall.")

        if start == end:
            return [start]

        # Priority queue stores (f_score, counter, position)
        # Counter ensures stable sorting when f_scores are equal
        open_set = []
        heapq.heappush(open_set, (self._heuristic(start, end), 0, start))
        
        g_score = {start: 0}
        came_from = {}
        closed_set = set()
        counter = 1

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == end:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dr, dc in directions:
                neighbor = (current[0] + dr, current[1] + dc)
                
                if not self._is_valid(neighbor) or self._is_wall(neighbor) or neighbor in closed_set:
                    continue

                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1

        return None


# =============================================================================
# Pytest Tests
# =============================================================================
import pytest

def test_simple_path():
    """Test basic pathfinding on an empty grid."""
    grid = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    assert len(path) == 5  # Manhattan distance + 1

def test_path_around_wall():
    """Test that path correctly avoids walls."""
    grid = [[1, 1, 1], [1, 0, 1], [1, 1, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert (1, 1) not in path  # Center is a wall

def test_weighted_optimal_path():
    """Test that A* chooses the lowest-cost path over shortest geometric path."""
    grid = [
        [1, 1, 1],
        [1, 100, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path is not None
    assert (1, 1) not in path  # High weight cell avoided
    # Verify optimality: total cost should be 4 (4 steps of weight 1)
    cost = sum(astar.grid[r][c] for r, c in path[1:])
    assert cost == 4

def test_start_equals_end():
    """Test edge case where start and end are the same."""
    grid = [[1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 0))
    assert path == [(0, 0)]

def test_out_of_bounds_raises_value_error():
    """Test that invalid coordinates raise ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_no_path_returns_none():
    """Test that unreachable targets return None."""
    grid = [[1, 0], [0, 1]]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (1, 1))
    assert path is None