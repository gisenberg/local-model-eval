import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """
    A* pathfinding implementation on a weighted 2D grid.
    
    Attributes:
        grid: 2D list where 0 represents an impassable wall and positive integers
              represent the movement cost to enter that cell.
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the AStarGrid with the provided cost grid.
        
        Args:
            grid: A 2D list of integers representing movement costs.
        """
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Calculate the Manhattan distance heuristic between two points.
        
        Args:
            a: Current coordinate (row, col).
            b: Target coordinate (row, col).
            
        Returns:
            Manhattan distance (|x1-x2| + |y1-y2|).
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid 4-directional neighbors for a given node.
        
        Args:
            node: Current coordinate (row, col).
            
        Returns:
            List of valid neighbor coordinates.
        """
        r, c = node
        neighbors = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # Up, Down, Left, Right

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                if self.grid[nr][nc] != 0: # Check if not a wall
                    neighbors.append((nr, nc))
        
        return neighbors

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the shortest path from start to end using A* algorithm.
        
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

        # Handle edge case: start is a wall or end is a wall
        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Handle edge case: start equals end
        if start == end:
            return [start]

        # Priority queue: (f_score, counter, node)
        # Counter is used to break ties in heap to ensure stability
        counter = 0
        open_set = []
        heapq.heappush(open_set, (0, counter, start))
        
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
        
        # g_score: cost from start to current node
        g_score: dict[Tuple[int, int], int] = {start: 0}
        
        # f_score: g_score + heuristic
        f_score: dict[Tuple[int, int], int] = {start: self._heuristic(start, end)}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == end:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._get_neighbors(current):
                # Cost to enter the neighbor cell
                tentative_g = g_score[current] + self.grid[neighbor[0]][neighbor[1]]

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    f_score[neighbor] = f
                    
                    counter += 1
                    heapq.heappush(open_set, (f, counter, neighbor))

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
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1] # Reverse to get start -> end
