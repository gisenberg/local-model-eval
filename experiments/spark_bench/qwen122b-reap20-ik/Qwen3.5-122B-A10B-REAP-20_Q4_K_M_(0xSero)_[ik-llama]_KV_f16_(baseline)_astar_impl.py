import heapq
from typing import List, Tuple, Optional
from math import inf

class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.
    
    Attributes:
        grid (List[List[int]]): The 2D grid where 0 represents a wall and 
                                positive integers represent movement costs.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with the provided grid.
        
        Args:
            grid: A 2D list of integers representing movement costs.
                  0 = impassable wall, >0 = cost to enter.
        """
        if not grid or not grid[0]:
            self.grid = []
            self.rows = 0
            self.cols = 0
            return

        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

        # Validate grid consistency (optional but good practice)
        for row in self.grid:
            if len(row) != self.cols:
                raise ValueError("Grid rows must have consistent column counts.")

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate the Manhattan distance heuristic between two points.
        
        Args:
            a: Start coordinate (row, col).
            b: End coordinate (row, col).
            
        Returns:
            The Manhattan distance (|r1-r2| + |c1-c2|).
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[Tuple[int, int], int]]:
        """
        Get valid neighbors for a given node in 4 directions (up, down, left, right).
        
        Args:
            node: Current coordinate (row, col).
            
        Returns:
            A list of tuples containing (neighbor_coords, movement_cost).
            Movement cost is the value of the neighbor cell in the grid.
        """
        r, c = node
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                cost = self.grid[nr][nc]
                if cost > 0: # 0 is a wall
                    neighbors.append(((nr, nc), cost))
        
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using the A* algorithm.
        
        Args:
            start: Starting coordinate (row, col).
            end: Ending coordinate (row, col).
            
        Returns:
            A list of coordinates representing the path from start to end (inclusive),
            or None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start coordinate {start} is out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End coordinate {end} is out of bounds.")

        # Edge case: Start is a wall or End is a wall
        if self.grid[start[0]][start[1]] == 0:
            return None
        if self.grid[end[0]][end[1]] == 0:
            return None

        # Edge case: Start equals End
        if start == end:
            return [start]

        # Priority queue: (f_score, counter, node)
        # Counter is used to break ties in heap to ensure stability
        open_set = []
        counter = 0
        
        # g_score: cost from start to current node
        g_score = {start: 0}
        
        # To reconstruct path
        came_from = {}

        # Initial f_score = g_score + heuristic
        h_start = self._heuristic(start, end)
        heapq.heappush(open_set, (h_start, counter, start))
        counter += 1

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                # Reconstruct path
                path = []
                curr = current
                while curr in came_from:
                    path.append(curr)
                    curr = came_from[curr]
                path.append(start)
                return path[::-1]

            for neighbor, move_cost in self._get_neighbors(current):
                tentative_g = g_score[current] + move_cost

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_score, counter, neighbor))
                    counter += 1

        return None
