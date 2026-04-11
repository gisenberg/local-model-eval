import heapq
from typing import List, Tuple, Optional, Set

class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.
    
    Attributes:
        grid (List[List[int]]): 2D list representing the grid where 0 is a wall 
                                and positive integers are movement costs.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with the provided 2D grid.
        
        Args:
            grid: A 2D list of integers representing movement costs.
                  0 represents an impassable wall.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty.")
        
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        
        # Validate that all rows have the same length
        for row in grid:
            if len(row) != self.cols:
                raise ValueError("All rows in the grid must have the same length.")

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate the Manhattan distance heuristic between two points.
        
        Args:
            a: Tuple (row, col) for the current node.
            b: Tuple (row, col) for the goal node.
            
        Returns:
            The Manhattan distance between a and b.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid 4-directional neighbors for a given position.
        
        Args:
            pos: Tuple (row, col) representing the current position.
            
        Returns:
            A list of valid (row, col) tuples for neighbors.
        """
        r, c = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right
        neighbors = []
        
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] != 0: # Check if not a wall
                    neighbors.append((nr, nc))
                    
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using the A* algorithm.
        
        Args:
            start: Tuple (row, col) representing the starting position.
            end: Tuple (row, col) representing the target position.
            
        Returns:
            A list of (row, col) tuples representing the path from start to end,
            inclusive. Returns None if no path exists.
            
        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        # Validate bounds
        if not (0 <= start[0] < self.rows and 0 <= start[1] < self.cols):
            raise ValueError(f"Start coordinates {start} are out of bounds.")
        if not (0 <= end[0] < self.rows and 0 <= end[1] < self.cols):
            raise ValueError(f"End coordinates {end} are out of bounds.")

        # Edge case: Start or End is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Edge case: Start equals End
        if start == end:
            return [start]

        # Priority queue: (f_score, counter, row, col)
        # Counter is used to break ties in heap to ensure stability
        counter = 0
        open_set = []
        heapq.heappush(open_set, (0, counter, start[0], start[1]))
        
        # Track visited nodes to avoid re-processing
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score: cost from start to current node
        g_score: dict[Tuple[int, int], int] = {start: 0}
        
        # f_score: g_score + heuristic
        f_score: dict[Tuple[int, int], int] = {start: self._heuristic(start, end)}
        
        open_set_hash: Set[Tuple[int, int]] = {start}

        while open_set:
            _, _, current_r, current_c = heapq.heappop(open_set)
            current = (current_r, current_c)
            
            # If we reached the goal, reconstruct path
            if current == end:
                return self._reconstruct_path(came_from, current)
            
            open_set_hash.discard(current)

            for neighbor in self._get_neighbors(current):
                # Cost to enter the neighbor cell
                move_cost = self.grid[neighbor[0]][neighbor[1]]
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    f_score[neighbor] = f
                    
                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f, counter, neighbor[0], neighbor[1]))
                        open_set_hash.add(neighbor)
        
        return None

    def _reconstruct_path(self, came_from: dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from start to current node using the came_from map.
        
        Args:
            came_from: Dictionary mapping each node to its predecessor.
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
