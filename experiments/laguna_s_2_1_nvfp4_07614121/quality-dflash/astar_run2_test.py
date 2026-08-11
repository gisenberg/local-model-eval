from typing import List, Tuple, Optional
import heapq
from dataclasses import dataclass
from dataclasses import field

@dataclass
class Node:
    """Represents a node in the grid for A* pathfinding."""
    x: int
    y: int
    g_cost: int = 0  # Cost from start to current node
    h_cost: int = 0  # Heuristic cost from current node to end
    parent: Optional['Node'] = None

    @property
    def f_cost(self) -> int:
        """Total cost (g + h)."""
        return self.g_cost + self.h_cost

    def __lt__(self, other: 'Node') -> bool:
        """Comparison for priority queue based on f_cost."""
        return self.f_cost < other.f_cost

class AStarGrid:
    """
    A* Pathfinding implementation on a weighted 2D grid.

    Grid values:
    - 0: Wall (impassable)
    - Positive integer: Movement cost (weight)

    Attributes:
        grid: 2D list representing the grid
        rows: Number of rows in grid
        cols: Number of columns in grid
    """

    def __init__(self, grid: List[List[int]]):
        """
        Initialize the A* grid.

        Args:
            grid: 2D list where 0 represents walls and positive integers represent movement costs

        Raises:
            ValueError: If grid is empty or not rectangular
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")

        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

        # Validate grid is rectangular
        for row in grid:
            if len(row) != self.cols:
                raise ValueError("Grid must be rectangular")

    def find_path(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal path from start to end using A* algorithm.

        Args:
            start: Starting position as (x, y) tuple
            end: Ending position as (x, y) tuple

        Returns:
            List of (x, y) tuples representing the path from start to end,
            or None if no path exists

        Raises:
            ValueError: If start or end positions are out of bounds or on walls
        """
        # Validate inputs
        self._validate_position(start, "Start")
        self._validate_position(end, "End")

        # Handle case where start equals end
        if start == end:
            return [start]

        # Initialize open and closed sets
        open_set = []
        closed_set = set()

        # Create start node
        start_node = Node(start[0], start[1])
        start_node.h_cost = self._heuristic(start, end)

        # Add start node to open set
        heapq.heappush(open_set, start_node)

        # Keep track of nodes in open set for efficient lookup
        open_set_lookup = {start}

        while open_set:
            # Get node with lowest f_cost
            current_node = heapq.heappop(open_set)
            open_set_lookup.remove((current_node.x, current_node.y))

            # Add to closed set
            closed_set.add((current_node.x, current_node.y))

            # Check if we've reached the end
            if (current_node.x, current_node.y) == end:
                return self._reconstruct_path(current_node)

            # Explore neighbors
            for neighbor_pos in self._get_neighbors(current_node.x, current_node.y):
                neighbor_x, neighbor_y = neighbor_pos

                # Skip if already in closed set
                if (neighbor_x, neighbor_y) in closed_set:
                    continue

                # Calculate tentative g_cost
                tentative_g = current_node.g_cost + self.grid[neighbor_y][neighbor_x]

                # Check if this path to neighbor is better than any previous one
                if (neighbor_x, neighbor_y) not in open_set_lookup:
                    # New node discovered
                    neighbor_node = Node(neighbor_x, neighbor_y)
                    neighbor_node.g_cost = tentative_g
                    neighbor_node.h_cost = self._heuristic((neighbor_x, neighbor_y), end)
                    neighbor_node.parent = current_node

                    heapq.heappush(open_set, neighbor_node)
                    open_set_lookup.add((neighbor_x, neighbor_y))
                elif tentative_g < self._get_node_from_open(open_set, neighbor_x, neighbor_y).g_cost:
                    # Found a better path to this neighbor
                    neighbor_node = self._get_node_from_open(open_set, neighbor_x, neighbor_y)
                    neighbor_node.g_cost = tentative_g
                    neighbor_node.parent = current_node

                    # Re-heapify since we modified the node
                    heapq.heapify(open_set)

        # No path found
        return None

    def _validate_position(self, pos: Tuple[int, int], name: str) -> None:
        """
        Validate that a position is within bounds and not a wall.

        Args:
            pos: Position to validate as (x, y) tuple
            name: Name of the position for error messages

        Raises:
            ValueError: If position is out of bounds or on a wall
        """
        x, y = pos

        if not (0 <= x < self.cols and 0 <= y < self.rows):
            raise ValueError(f"{name} position ({x}, {y}) is out of bounds")

        if self.grid[y][x] == 0:
            raise ValueError(f"{name} position ({x}, {y}) is on a wall")

    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """
        Calculate Manhattan distance heuristic between two positions.

        Args:
            pos1: First position as (x, y) tuple
            pos2: Second position as (x, y) tuple

        Returns:
            Manhattan distance between positions
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_neighbors(self, x: int, y: int) -> List[Tuple[int, int]]:
        """
        Get valid neighboring positions (4-directional).

        Args:
            x: X coordinate
            y: Y coordinate

        Returns:
            List of valid neighbor positions as (x, y) tuples
        """
        neighbors = []
        # 4-directional movement: up, right, down, left
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy

            # Check bounds
            if 0 <= new_x < self.cols and 0 <= new_y < self.rows:
                # Check if not a wall
                if self.grid[new_y][new_x] != 0:
                    neighbors.append((new_x, new_y))

        return neighbors

    def _reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """
        Reconstruct path from start to end by following parent pointers.

        Args:
            node: End node to trace back from

        Returns:
            List of positions from start to end
        """
        path = []
        current = node

        while current is not None:
            path.append((current.x, current.y))
            current = current.parent

        # Reverse to get path from start to end
        return path[::-1]

    def _get_node_from_open(self, open_set: List[Node], x: int, y: int) -> Node:
        """
        Get a node from the open set by coordinates.

        Args:
            open_set: List of nodes in open set
            x: X coordinate
            y: Y coordinate

        Returns:
            Node at given coordinates
        """
        for node in open_set:
            if node.x == x and node.y == y:
                return node
        raise ValueError(f"Node ({x}, {y}) not found in open set")


# Tests
import pytest

def test_start_equals_end():
    """Test pathfinding when start and end are the same."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (0, 0))
    assert result == [(0, 0)]

def test_simple_path():
    """Test a simple 2x2 path."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (1, 1))
    assert result == [(0, 0), (1, 1)] or result == [(0, 0), (0, 1), (1, 1)]

def test_wall_blocking():
    """Test pathfinding when walls block direct path."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (2, 0))
    assert result == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]

def test_no_path():
    """Test when no path exists due to wall blocking."""
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1]
    ]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (2, 0))
    assert result is None

def test_out_of_bounds():
    """Test that out of bounds raises ValueError."""
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)

    with pytest.raises(ValueError):
        astar.find_path((-1, 0), (1, 1))

    with pytest.raises(ValueError):
        astar.find_path((0, 0), (2, 2))

def test_weighted_path():
    """Test that algorithm finds optimal path with weighted cells."""
    grid = [
        [1, 1, 1],
        [1, 9, 1],
        [1, 1, 1]
    ]
    astar = AStarGrid(grid)
    result = astar.find_path((0, 0), (2, 0))

    # Should go around the high-cost cell
    expected_paths = [
        [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0)],
        [(0, 0), (1, 0), (2, 0)]
    ]
    assert result in expected_paths