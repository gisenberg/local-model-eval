import heapq
from typing import List, Tuple, Optional, Set
import pytest

class AStarGrid:
    """
    A class to implement A* pathfinding on a weighted 2D grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with a 2D grid of movement costs.
        
        Args:
            grid: A 2D list where 0 represents a wall and positive integers 
                  represent the cost to enter that cell.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two points.
        
        Args:
            a: First coordinate (row, col).
            b: Second coordinate (row, col).
            
        Returns:
            Manhattan distance.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
        Args:
            start: Starting coordinates (row, col).
            end: Ending coordinates (row, col).
            
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

        # Handle start == end edge case
        if start == end:
            return [start]

        # Check if start or end is a wall (cost 0)
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None

        # Directions: Up, Down, Left, Right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Priority Queue: (f_score, counter, row, col)
        # Counter is used to break ties in heap to avoid comparing tuples (r, c)
        open_set = []
        counter = 0
        heapq.heappush(open_set, (0, counter, start[0], start[1]))

        # Track g_score (cost from start to current node)
        g_score: dict[Tuple[int, int], int] = {start: 0}
        
        # Track parent nodes for path reconstruction
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}

        # Set to keep track of visited nodes (closed set)
        closed_set: Set[Tuple[int, int]] = set()

        while open_set:
            _, _, current_r, current_c = heapq.heappop(open_set)
            current = (current_r, current_c)

            if current in closed_set:
                continue
            
            closed_set.add(current)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for dr, dc in directions:
                neighbor_r, neighbor_c = current_r + dr, current_c + dc
                neighbor = (neighbor_r, neighbor_c)

                if not self._is_valid(neighbor_r, neighbor_c):
                    continue
                
                # Skip walls
                if self.grid[neighbor_r][neighbor_c] == 0:
                    continue

                # Cost to enter the neighbor cell
                move_cost = self.grid[neighbor_r][neighbor_c]
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, counter, neighbor_r, neighbor_c))
                    counter += 1
                    came_from[neighbor] = current

        return None

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from end to start using the came_from map.
        
        Args:
            came_from: Dictionary mapping nodes to their predecessors.
            current: The end node.
            
        Returns:
            List of coordinates from start to end.
        """
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        total_path.reverse()
        return total_path


# --- Pytest Tests ---

def test_simple_path_uniform_grid():
    """Test 1: Simple path on a uniform grid (all costs are 1)."""
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    assert path[0] == (0, 0)
    assert path[-1] == (2, 2)
    # Optimal path length in steps is 4 (Manhattan distance), cost is 4 (entering 4 cells)
    # Path: (0,0)->(0,1)->(0,2)->(1,2)->(2,2) or similar
    assert len(path) == 5  # 5 nodes in path
    total_cost = sum(grid[r][c] for r, c in path[1:]) # Exclude start cost
    assert total_cost == 4

def test_path_around_obstacles():
    """Test 2: Path finding around obstacles (0s)."""
    grid = [
        [1, 1, 1, 1],
        [1, 0, 0, 1],
        [1, 1, 1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 3))
    
    assert path is not None
    # Verify no walls in path
    for r, c in path:
        assert grid[r][c] != 0
    # Verify connectivity
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        assert abs(r1 - r2) + abs(c1 - c2) == 1

def test_weighted_grid_optimality():
    """Test 3: Weighted grid where path prefers lower-cost cells."""
    # Direct path has high cost, detour has lower cost
    grid = [
        [1, 1, 1],
        [1, 10, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    # Start (0,0) to End (2,2)
    # Direct path through (1,1) cost: 1 + 10 + 1 = 12
    # Detour path (0,0)->(0,1)->(0,2)->(1,2)->(2,2) cost: 1+1+1+1 = 4
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is not None
    total_cost = sum(grid[r][c] for r, c in path[1:])
    assert total_cost == 4  # Should avoid the 10 cost cell

def test_no_path_exists():
    """Test 4: No path exists (fully blocked)."""
    grid = [
        [1, 0, 1],
        [0, 0, 0],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    
    assert path is None

def test_start_equals_end():
    """Test 5: Start coordinates equal end coordinates."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 1), (1, 1))
    
    assert path is not None
    assert path == [(1, 1)]

def test_invalid_coordinates():
    """Test 6: Invalid coordinates raise ValueError."""
    grid = [
        [1, 1],
        [1, 1]
    ]
    astar = AStarGrid(grid)
    
    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (0, 0))
        
    with pytest.raises(ValueError):
        astar.find_path((0, 0), (10, 10))

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
