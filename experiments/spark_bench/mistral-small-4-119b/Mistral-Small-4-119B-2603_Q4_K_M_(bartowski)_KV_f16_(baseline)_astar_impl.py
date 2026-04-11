import heapq
from typing import List, Tuple, Optional, Dict, Set

class AStarGrid:
    """
    A class to perform A* pathfinding on a weighted 2D grid.

    The grid values represent movement costs (0 = impassable wall, positive int = cost to enter that cell).
    Supports 4-directional movement (up, down, left, right).
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid with movement costs.

        Args:
            grid: A 2D list where each element represents the movement cost of that cell.
                 0 indicates an impassable wall.

        Raises:
            ValueError: If the grid is empty or rows have inconsistent lengths.
        """
        if not grid or not all(len(row) == len(grid[0]) for row in grid):
            raise ValueError("Grid must be non-empty and have consistent row lengths")

        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if self.rows > 0 else 0

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start: Tuple of (row, col) coordinates for the starting position.
            end: Tuple of (row, col) coordinates for the target position.

        Returns:
            List of (row, col) coordinates representing the path from start to end (inclusive),
            or None if no path exists.

        Raises:
            ValueError: If start or end coordinates are out of bounds.
        """
        if not self._is_valid_coordinate(start) or not self._is_valid_coordinate(end):
            raise ValueError("Start or end coordinates are out of bounds")

        if start == end:
            return [start]

        if self.grid[start[0]][start[1]] == 0 or self.grid[end[0]][end[1]] == 0:
            return None

        # Priority queue for open set: (f_score, g_score, current_node, path)
        open_set = []
        heapq.heappush(open_set, (0, 0, start, [start]))

        # For tracking the best path to each node
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}

        # Cost from start to each node
        g_score: Dict[Tuple[int, int], float] = {start: 0}

        # Set of visited nodes
        visited: Set[Tuple[int, int]] = set()

        while open_set:
            _, current_g, current_node, path = heapq.heappop(open_set)

            if current_node == end:
                return path

            if current_node in visited:
                continue

            visited.add(current_node)

            for neighbor in self._get_neighbors(current_node):
                if neighbor in visited:
                    continue

                tentative_g = current_g + self.grid[neighbor[0]][neighbor[1]]

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._manhattan_distance(neighbor, end)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, path + [neighbor]))
                    came_from[neighbor] = current_node

        return None

    def _is_valid_coordinate(self, coord: Tuple[int, int]) -> bool:
        """
        Check if a coordinate is within the grid bounds.

        Args:
            coord: Tuple of (row, col) coordinates to check.

        Returns:
            True if the coordinate is valid, False otherwise.
        """
        row, col = coord
        return 0 <= row < self.rows and 0 <= col < self.cols

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighboring nodes (up, down, left, right) that are not walls.

        Args:
            node: Tuple of (row, col) coordinates for the current node.

        Returns:
            List of valid neighboring nodes.
        """
        row, col = node
        neighbors = [
            (row + 1, col),
            (row - 1, col),
            (row, col + 1),
            (row, col - 1)
        ]

        valid_neighbors = []
        for neighbor in neighbors:
            if self._is_valid_coordinate(neighbor) and self.grid[neighbor[0]][neighbor[1]] != 0:
                valid_neighbors.append(neighbor)

        return valid_neighbors

    def _manhattan_distance(self, node1: Tuple[int, int], node2: Tuple[int, int]) -> int:
        """
        Calculate the Manhattan distance between two nodes.

        Args:
            node1: Tuple of (row, col) coordinates for the first node.
            node2: Tuple of (row, col) coordinates for the second node.

        Returns:
            Manhattan distance between node1 and node2.
        """
        return abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])
