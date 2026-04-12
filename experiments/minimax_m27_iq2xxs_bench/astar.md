# MiniMax-M2.7 UD-IQ2_XXS (empty-think, 192K ctx) — A* Pathfinding

## Output (13039 chars)

Here's a full implementation that satisfies all the requirements:

```python
"""
A* Pathfinding on a Weighted 2D Grid

This module provides an A* search implementation for finding optimal (minimum-cost)
paths on a 2D grid with weighted cells. Movement is 4-directional (no diagonals).
"""

from __future__ import annotations

import heapq
from typing import List, Optional, Tuple

import pytest


class AStarGrid:
    """
    A* pathfinding implementation for a weighted 2D grid.

    The grid is a 2D list where each cell value represents the cost to enter that cell.
    A value of 0 indicates an impassable wall. The A* algorithm uses a Manhattan
    distance heuristic and a min-heap (priority queue) to explore nodes.

    Attributes:
        grid (List[List[int]]): The 2D grid of movement costs.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
    """

    def __init__(self, grid: List[List[int]]) -> None:
        """
        Initialize the A* grid pathfinder.

        Args:
            grid: A 2D list of integers where each value is the movement cost
                  for that cell. A value of 0 means the cell is impassable.

        Raises:
            ValueError: If the grid is empty or not rectangular.
        """
        if not grid or not grid[0]:
            raise ValueError("Grid cannot be empty")
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])
        # Validate that all rows have the same length
        for row in self.grid:
            if len(row) != self.cols:
                raise ValueError("All rows must have the same length")

    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        """
        Compute the Manhattan distance between two points.

        Manhattan distance is the sum of absolute differences in row and column.

        Args:
            a: First point as (row, col).
            b: Second point as (row, col).

        Returns:
            The Manhattan distance between points a and b.
        """
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """
        Check if a position is within grid bounds and not a wall.

        Args:
            pos: Position as (row, col).

        Returns:
            True if the position is valid (in bounds and not a wall), False otherwise.
        """
        row, col = pos
        return 0 <= row < self.rows and 0 <= col < self.cols and self.grid[row][col] > 0

    def find_path(
        self, start: Tuple[int, int], end: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find the optimal (minimum cost) path from start to end using A*.

        Args:
            start: Starting position as (row, col).
            end: Goal position as (row, col).

        Returns:
            A list of positions from start to end inclusive representing the optimal
            path, or None if no path exists.

        Raises:
            ValueError: If start or end is out of bounds or is a wall.
        """
        # Validate start position
        if not self._is_valid(start):
            raise ValueError(f"Start position {start} is invalid or a wall")
        # Validate end position
        if not self._is_valid(end):
            raise ValueError(f"End position {end} is invalid or a wall")

        # Edge case: start equals end
        if start == end:
            return [start]

        # Directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # open_set: min-heap of (f_score, counter, position)
        # counter ensures consistent ordering for heapq (tie-breaking)
        counter = 0
        heap: List[Tuple[int, int, Tuple[int, int]]] = []

        # g_score[pos] = cost from start to pos
        g_score = {start: self.grid[start[0]][start[1]]}
        # f_score[pos] = g_score[pos] + heuristic(pos, end)
        f_score = {start: g_score[start] + self._manhattan_distance(start, end)}

        heapq.heappush(heap, (f_score[start], counter, start))
        counter += 1

        # came_from[pos] = the position we came from to reach pos (for path reconstruction)
        came_from: dict[Tuple[int, int], Tuple[int, int]] = {}

        # Set of positions we've fully evaluated (closed set)
        closed_set: set[Tuple[int, int]] = set()

        while heap:
            # Pop the node with lowest f_score
            _, _, current = heapq.heappop(heap)

            # If we've reached the goal, reconstruct and return the path
            if current == end:
                return self._reconstruct_path(came_from, start, end)

            # Mark current as fully evaluated
            closed_set.add(current)

            row, col = current
            for dr, dc in directions:
                neighbor = (row + dr, col + dc)

                # Skip invalid neighbors
                if not self._is_valid(neighbor):
                    continue
                # Skip neighbors already fully evaluated
                if neighbor in closed_set:
                    continue

                # Cost to enter neighbor
                cost = self.grid[neighbor[0]][neighbor[1]]
                # Tentative g_score for neighbor going through current
                tentative_g = g_score[current] + cost

                # If this path to neighbor is better than any previous path
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    # Record this best path
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._manhattan_distance(neighbor, end)
                    heapq.heappush(heap, (f_score[neighbor], counter, neighbor))
                    counter += 1

        # No path found
        return None

    def _reconstruct_path(
        self,
        came_from: dict[Tuple[int, int], Tuple[int, int]],
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Reconstruct the path from start to end using the came_from map.

        Args:
            came_from: Dictionary mapping each position to its predecessor.
            start: Starting position.
            end: Goal position.

        Returns:
            List of positions from start to end.
        """
        path = [end]
        current = end
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def compute_total_cost(self, path: List[Tuple[int, int]]) -> int:
        """
        Compute the total movement cost of a given path.

        Args:
            path: List of positions representing a path.

        Returns:
            Total cost of traversing the path.

        Raises:
            ValueError: If the path is empty or invalid.
        """
        if not path:
            raise ValueError("Path cannot be empty")
        total = 0
        for pos in path:
            row, col = pos
            if not (0 <= row < self.rows and 0 <= col < self.cols):
                raise ValueError(f"Position {pos} is out of bounds")
            total += self.grid[row][col]
        return total


# --- Pytest Tests ---


class TestAStarGrid:
    """Test suite for AStarGrid pathfinding."""

    def test_simple_path_on_uniform_grid(self) -> None:
        """
        Test basic pathfinding on a uniform grid with no obstacles.
        """
        grid = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (4, 4))

        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)

        # Verify path is contiguous (each step is adjacent)
        for i in range(len(path) - 1):
            row_diff = abs(path[i][0] - path[i + 1][0])
            col_diff = abs(path[i][1] - path[i + 1][1])
            assert row_diff + col_diff == 1, f"Path steps must be adjacent: {path[i]} -> {path[i + 1]}"

        # Verify total cost is optimal (Manhattan distance * 1 cost)
        total_cost = astar.compute_total_cost(path)
        expected_cost = 9  # 8 steps + cost of start cell
        assert total_cost == expected_cost, f"Expected cost {expected_cost}, got {total_cost}"

    def test_path_around_obstacles(self) -> None:
        """
        Test pathfinding with obstacles blocking the direct route.
        """
        grid = [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (4, 4))

        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)

        # Verify path avoids walls
        for pos in path:
            assert grid[pos[0]][pos[1]] > 0, f"Path includes wall at {pos}"

        # Verify total cost is optimal
        total_cost = astar.compute_total_cost(path)
        # Optimal path: (0,0)->(0,1)->(0,2)->(0,3)->(0,4)->(1,4)->(2,4)->(3,4)->(4,4)
        expected_cost = 9
        assert total_cost == expected_cost, f"Expected cost {expected_cost}, got {total_cost}"

    def test_weighted_grid_prefers_lower_cost(self) -> None:
        """
        Test that the path prefers lower-cost cells when given a choice.
        """
        grid = [
            [1, 10, 1, 1, 1],
            [1, 10, 1, 1, 1],
            [1, 10, 1, 1, 1],
            [1, 10, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (4, 4))

        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (4, 4)

        # Verify path avoids the expensive column (col 1)
        for pos in path:
            assert pos[1] != 1, f"Path should avoid expensive column 1: found {pos}"

        # Verify total cost is optimal
        total_cost = astar.compute_total_cost(path)
        # Optimal path goes through column 0, then row 4: 5+4=9 cells at cost 1 each
        expected_cost = 9
        assert total_cost == expected_cost, f"Expected cost {expected_cost}, got {total_cost}"

    def test_no_path_exists_fully_blocked(self) -> None:
        """
        Test that None is returned when no path exists due to full blockage.
        """
        grid = [
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1],
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (4, 4))

        assert path is None

    def test_start_equals_end(self) -> None:
        """
        Test that start == end returns [start].
        """
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((1, 1), (1, 1))

        assert path is not None
        assert path == [(1, 1)]
        assert len(path) == 1

    def test_invalid_coordinates(self) -> None:
        """
        Test that ValueError is raised for out-of-bounds or wall positions.
        """
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
        astar = AStarGrid(grid)

        # Test out-of-bounds start
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (2, 2))

        # Test out-of-bounds end
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (3, 3))

        # Test wall start
        with pytest.raises(ValueError):
            astar.find_path((1, 1), (2, 2))

        # Test wall end
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (1, 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## Key Implementation Details

### 1. **A* Algorithm Components**
   - **Open set**: Min-heap (`heapq`) prioritized by `f_score = g_score + heuristic`
   - **Closed set**: Track fully evaluated nodes to avoid revisiting
   - **Came-from map**: For path reconstruction

### 2. **Heuristic**
   - Manhattan distance: `|row1 - row2| + |col1 - col2|`
   - Admissible (never overestimates) for 4-directional movement

### 3. **Edge Cases Handled**
   - `start == end`: Returns `[start]`
   - Wall positions: Raises `ValueError`
   - Out-of-bounds: Raises `ValueError`
   - No path exists: Returns `None`

### 4. **Path Validation**
   - Contiguity check (each step is adjacent)
   - Wall avoidance verification
   - Total cost computation

### 5. **Test Coverage**
   - Simple uniform grid path
   - Obstacle navigation
   - Weighted grid (prefers lower-cost paths)
   - Fully blocked grid (no path)
   - Start equals end
   - Invalid coordinates (out-of-bounds, walls)
