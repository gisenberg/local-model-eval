# Nemotron 3 Nano 4B Q4_K_M — astar

## Output

```python
import heapq
from typing import List, Tuple, Optional, Dict, Any
from heapq import heappush, heappop


class AStarGrid:
    """
    A* pathfinding algorithm for a 2D grid with weighted cells.
    
    The grid is represented as a list of lists where each cell contains:
    - 0: impassable wall
    - positive int: movement cost to enter that cell
    
    A* uses Manhattan distance as heuristic for 4-directional movement.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the grid.
        
        Args:
            grid: 2D list representing the grid. 0 = wall, positive int = cost.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.visited: Dict[Tuple[int, int], bool] = {}
        self.cost: Dict[Tuple[int, int], int] = {}
        self.parent: Dict[Tuple[int, int], Tuple[int, int]] = {}

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A*.
        
        Args:
            start: (row, col) tuple representing the starting position
            end: (row, col) tuple representing the destination position
            
        Returns:
            List of (row, col) tuples representing the path from start to end, or None if no path exists
            
        Raises:
            ValueError: If start or end is out of bounds, or if either is a wall
        """
        # Validate inputs
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError("Start position is out of bounds")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError("End position is out of bounds")
        
        if self.grid[start[0]][start[1]] == 0:
            raise ValueError("Start position is a wall")
        if self.grid[end[0]][end[1]] == 0:
            raise ValueError("End position is a wall")
        
        # If start equals end
        if start == end:
            return [start]
        
        # Initialize data structures
        self.visited.clear()
        self.cost.clear()
        self.parent.clear()
        
        # Priority queue: (f_score, row, col)
        open_set: List[Tuple[int, int, int]] = []
        heappush(open_set, (self.heuristic(start, end), start[0], start[1]))
        
        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        while open_set:
            f_score, r, c = heappop(open_set)
            
            # Skip if already visited with lower cost
            if (r, c) in self.visited and self.cost[(r, c)] < f_score:
                continue
            
            # Mark as visited
            self.visited[(r, c)] = True
            self.cost[(r, c)] = f_score
            self.parent[(r, c)] = None
            
            # If we reached the end
            if (r, c) == end:
                return self._reconstruct_path(start, end)
            
            # Explore neighbors
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                
                # Check bounds
                if not (0 <= nr < self.rows and 0 <= nc < self.cols):
                    continue
                
                # Check if wall
                if self.grid[nr][nc] == 0:
                    continue
                
                # Calculate new cost
                new_cost = self.cost.get((r, c), 0) + self.grid[nr][nc]
                
                # Calculate f_score
                f = new_cost + self.heuristic((nr, nc), end)
                
                # If this is the first time we've seen this cell or we found a better path
                if (nr, nc) not in self.cost or new_cost < self.cost[(nr, nc)]:
                    self.cost[(nr, nc)] = new_cost
                    self.parent[(nr, nc)] = (r, c)
                    heappush(open_set, (f, nr, nc))
        
        # No path found
        return None

    def heuristic(self, node: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """
        Manhattan distance heuristic for 4-directional movement.
        
        Args:
            node: (row, col) tuple
            goal: (row, col) tuple
            
        Returns:
            Manhattan distance between node and goal
        """
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    def _reconstruct_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from start to end using parent pointers.
        
        Args:
            start: (row, col) tuple
            end: (row, col) tuple
            
        Returns:
            List of (row, col) tuples representing the path from start to end
        """
        path = []
        current = end
        while current != start:
            path.append(current)
            current = self.parent[current]
        path.append(start)
        path.reverse()
        return path


# Test cases
def test_start_equals_end():
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    grid_obj = AStarGrid(grid)
    path = grid_obj.find_path((1, 1), (1, 1))
    assert path == [(1, 1)], f"Expected [(1, 1)], got {path}"

def test_simple_path():
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    grid_obj = AStarGrid(grid)
    path = grid_obj.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)], f"Expected path, got {path}"

def test_wall_blocked():
    grid = [[1, 0, 1],
            [1, 1, 1],
            [1, 1, 1]]
    grid_obj = AStarGrid(grid)
    path = grid_obj.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)], f"Expected path, got {path}"

def test_out_of_bounds():
    grid = [[1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    grid_obj = AStarGrid(grid)
    try:
        grid_obj.find_path((-1, 0), (0, 0))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_start_wall():
    grid = [[0, 1, 1],
            [1, 1, 1],
            [1, 1, 1]]
    grid_obj = AStarGrid(grid)
    try:
        grid_obj.find_path((0, 0), (2, 2))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_complex_path():
    grid = [[1, 2, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1]]
    grid_obj = AStarGrid(grid)
    path = grid_obj.find_path((0, 0), (3, 3))
    assert path == [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3)], f"Expected path, got {path}"

if __name__ == "__main__":
    # Run tests
    test_start_equals_end()
    test_simple_path()
    test_wall_blocked()
    test_out_of_bounds()
    test_start_wall()
    test_complex_path()
    print("All tests passed!")
```
