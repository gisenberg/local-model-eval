import heapq
from typing import List, Tuple, Optional, Set

class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.
    
    Attributes:
        grid: A 2D list where 0 represents an impassable wall and positive integers
              represent the movement cost to enter that cell.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid with the provided cost map.
        
        Args:
            grid: 2D list of integers representing movement costs.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # Validate grid consistency (optional but good practice)
        for row in grid:
            if len(row) != self.cols:
                raise ValueError("Grid rows must have consistent length.")

    def _is_valid(self, r: int, c: int) -> bool:
        """Check if coordinates are within grid bounds."""
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _is_wall(self, r: int, c: int) -> bool:
        """Check if a cell is a wall (cost 0)."""
        return self.grid[r][c] == 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance between two points.
        
        Args:
            a: Start coordinate (row, col).
            b: End coordinate (row, col).
            
        Returns:
            Manhattan distance.
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
        # Validate bounds
        if not self._is_valid(*start):
            raise ValueError(f"Start coordinates {start} are out of bounds.")
        if not self._is_valid(*end):
            raise ValueError(f"End coordinates {end} are out of bounds.")

        # Edge case: Start or End is a wall
        if self._is_wall(*start) or self._is_wall(*end):
            return None

        # Edge case: Start equals End
        if start == end:
            return [start]

        # Directions: Up, Down, Left, Right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Priority Queue: (f_score, g_score, row, col)
        # We use a counter to break ties if f_scores are equal, ensuring stability
        # though not strictly required for correctness.
        open_set = []
        heapq.heappush(open_set, (0, 0, start[0], start[1]))

        # Track visited nodes and their best g_scores
        g_scores: dict[Tuple[int, int], int] = {start: 0}
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}

        while open_set:
            _, current_g, r, c = heapq.heappop(open_set)
            current = (r, c)

            # If we reached the goal, reconstruct path
            if current == end:
                return self._reconstruct_path(came_from, current)

            # Skip if we found a better path to this node already
            if current_g > g_scores.get(current, float('inf')):
                continue

            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                neighbor = (nr, nc)

                if not self._is_valid(nr, nc) or self._is_wall(nr, nc):
                    continue

                # Cost to enter the neighbor cell
                move_cost = self.grid[nr][nc]
                tentative_g = current_g + move_cost

                if tentative_g < g_scores.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, nr, nc))

        return None

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from end to start using the came_from map.
        
        Args:
            came_from: Dictionary mapping each node to its predecessor.
            current: The end node.
            
        Returns:
            List of coordinates from start to end.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def calculate_path_cost(self, path: List[Tuple[int, int]]) -> int:
        """
        Calculate the total cost of a given path.
        Note: The cost includes the entry cost of every cell in the path except the start.
        
        Args:
            path: List of (row, col) tuples.
            
        Returns:
            Total movement cost.
        """
        if not path:
            return 0
        
        total_cost = 0
        # Start node cost is usually not counted as "movement" into it, 
        # but based on the problem description "cost to enter that cell",
        # we sum the costs of all cells except the start.
        for r, c in path[1:]:
            total_cost += self.grid[r][c]
        return total_cost
