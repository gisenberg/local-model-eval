import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """A* pathfinding on a weighted 2D grid.

    The grid is a 2D list where:
    - 0 represents a wall (impassable).
    - Positive integers represent the cost of moving into that cell.

    Movement is 4-directional (up, down, left, right).
    The Manhattan distance heuristic is used.
    """

    DIRECTION_DELTA: List[Tuple[int, int]] = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(self, grid: List[List[int]]) -> None:
        """Initialise the AStarGrid.

        Args:
            grid: 2D list of integers where 0 is a wall and positive values
                  are the cost of moving into that cell.
        """
        self.grid: List[List[int]] = grid
        self.rows: int = len(grid)
        self.cols: int = len(grid[0]) if self.rows > 0 else 0

    def _in_bounds(self, pos: Tuple[int, int]) -> bool:
        """Return True if *pos* is inside the grid."""
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    @staticmethod
    def _heuristic(pos: Tuple[int, int], goal: Tuple[int, int]) -> int:
        """Manhattan distance from *pos* to *goal*."""
        return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

    def find_path(
        self,
        start: Tuple[int, int],
        end: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """Find the optimal (lowest-cost) path from *start* to *end*.

        The cost of a step equals the weight of the cell being moved **into**.

        Args:
            start: (row, col) tuple for the starting position.
            end:   (row, col) tuple for the ending position.

        Returns:
            A list of (row, col) tuples from start to end (inclusive),
            or None if no path exists.

        Raises:
            ValueError: If *start* or *end* is out of bounds.
        """
        # --- preconditions ---
        if not self._in_bounds(start):
            raise ValueError(f"Start position {start} is out of bounds.")
        if not self._in_bounds(end):
            raise ValueError(f"End position {end} is out of bounds.")

        if self.grid[start[0]][start[1]] == 0:
            return None  # start is a wall
        if self.grid[end[0]][end[1]] == 0:
            return None  # end is a wall

        if start == end:
            return [start]

        # --- A* bookkeeping ---
        # (f_score, tie_break, position)
        tie_counter = 0
        open_heap: List[Tuple[int, int, Tuple[int, int]]] = [
            (self._heuristic(start, end), 0, start)
        ]

        g_score: dict = {start: 0}          # lowest known g-score
        came_from: dict = {}                 # predecessor map
        closed_set: set = set()              # already expanded

        while open_heap:
            _, _, current = heapq.heappop(open_heap)

            if current in closed_set:
                continue
            closed_set.add(current)

            if current == end:
                # reconstruct path
                path: List[Tuple[int, int]] = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return list(reversed(path))

            current_g = g_score[current]

            for dr, dc in self.DIRECTION_DELTA:
                neighbour = (current[0] + dr, current[1] + dc)

                if not self._in_bounds(neighbour):
                    continue
                if neighbour in closed_set:
                    continue
                if self.grid[neighbour[0]][neighbour[1]] == 0:
                    continue

                move_cost = self.grid[neighbour[0]][neighbour[1]]
                tentative_g = current_g + move_cost

                if tentative_g < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g
                    f = tentative_g + self._heuristic(neighbour, end)
                    tie_counter += 1
                    heapq.heappush(open_heap, (f, tie_counter, neighbour))

        return None  # no path found

"""Tests for AStarGrid – run with:  pytest test_astar.py -v"""

import pytest

# ---------------------------------------------------------------------------
# 1. Simple path on a uniform grid
# ---------------------------------------------------------------------------
def test_simple_path() -> None:
    grid = [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (2, 2))
    assert path == [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
    # Cost = 1 + 1 + 1 + 1 = 4  (4 steps into weight-1 cells)

# ---------------------------------------------------------------------------
# 2. Start equals end
# ---------------------------------------------------------------------------
def test_start_equals_end() -> None:
    grid = [[5, 1], [1, 3]]
    astar = AStarGrid(grid)
    path = astar.find_path((1, 0), (1, 0))
    assert path == [(1, 0)]

# ---------------------------------------------------------------------------
# 3. No path exists (walls block all routes)
# ---------------------------------------------------------------------------
def test_no_path_walls() -> None:
    grid = [
        [1, 0, 1],
        [1, 0, 1],
        [1, 0, 1],
    ]
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path is None

# ---------------------------------------------------------------------------
# 4. Out-of-bounds start raises ValueError
# ---------------------------------------------------------------------------
def test_out_of_bounds_start() -> None:
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="Start position"):
        astar.find_path((3, 0), (0, 0))

# ---------------------------------------------------------------------------
# 5. Out-of-bounds end raises ValueError
# ---------------------------------------------------------------------------
def test_out_of_bounds_end() -> None:
    grid = [[1, 1], [1, 1]]
    astar = AStarGrid(grid)
    with pytest.raises(ValueError, match="End position"):
        astar.find_path((0, 0), (-1, 1))

# ---------------------------------------------------------------------------
# 6. Weighted optimal path – longer geometrically but cheaper total cost
# ---------------------------------------------------------------------------
def test_weighted_optimal_path() -> None:
    """A* must choose the cheap-but-long route over the expensive shortcut."""
    grid = [
        [1, 10, 1],
        [1, 10, 1],
        [1,  1, 1],
    ]
    # Shortcut  (0,0)->(0,1)->(0,2)  cost = 10 + 1 = 11
    # Long way  (0,0)->(1,0)->(2,0)->(2,1)->(2,2)->(1,2)->(0,2)  cost = 6
    astar = AStarGrid(grid)
    path = astar.find_path((0, 0), (0, 2))
    assert path == [
        (0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2), (0, 2),
    ]