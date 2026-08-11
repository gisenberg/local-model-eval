```python
# astar_grid.py
import heapq
from typing import List, Tuple, Optional

class AStarGrid:
    """A* pathfinder on a 4-connected weighted grid."""
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.rows and 0 <= c < self.cols

    def _passable(self, r: int, c: int) -> bool:
        return self.grid[r][c] != 0

    def _heuristic(self, a: Tuple[int,int], b: Tuple[int,int]) -> int:
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def find_path(self, start: Tuple[int,int], end: Tuple[int,int]) -> Optional[List[Tuple[int,int]]]:
        """Return cheapest path inclusive of start/end or None."""
        sr, sc = start
        er, ec = end

        if not self._in_bounds(sr, sc) or not self._in_bounds(er, ec):
            raise ValueError("start or end out of bounds")

        if not self._passable(sr, sc) or not self._passable(er, ec):
            return None

        if start == end:
            return [start]

        open_heap = []
        heapq.heappush(open_heap, (self._heuristic(start, end), 0, start))
        g_score = {start: 0}
        came_from = {}
        closed = set()

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == end:
                # reconstruct
                path = [end]
                while path[-1] != start:
                    path.append(came_from[path[-1]])
                return list(reversed(path))

            closed.add(current)
            cr, cc = current
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nr, nc = cr+dr, cc+dc
                if not self._in_bounds(nr, nc) or not self._passable(nr, nc):
                    continue
                neighbor = (nr, nc)
                tentative_g = g_score[current] + self.grid[nr][nc]  # cost to enter
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, end)
                    heapq.heappush(open_heap, (f, tentative_g, neighbor))
        return None
```

```python
# test_astar_grid.py
import pytest
from astar_grid import AStarGrid

def path_cost(grid, path):
    return sum(grid[r][c] for r,c in path[1:])  # start not counted

def test_simple_path():
    grid = [[1,1,1],[1,1,1],[1,1,1]]
    a = AStarGrid(grid)
    p = a.find_path((0,0),(2,2))
    assert p is not None
    assert p[0]==(0,0) and p[-1]==(2,2)
    assert len(p)==5
    assert path_cost(grid,p)==4

def test_path_around_obstacles():
    grid = [
        [1,1,1],
        [0,0,1],
        [1,1,1]
    ]
    a = AStarGrid(grid)
    p = a.find_path((0,0),(2,0))
    assert p is not None
    assert p[0]==(0,0) and p[-1]==(2,0)
    # must go via top right
    for r,c in p:
        assert grid[r][c]!=0
    # valid adjacency
    for (r1,c1),(r2,c2) in zip(p,p[1:]):
        assert abs(r1-r2)+abs(c1-c2)==1
    assert path_cost(grid,p)==4

def test_weighted_grid_prefers_lower_cost():
    grid = [
        [1,100,1],
        [1,100,1],
        [1,1,1]
    ]
    a = AStarGrid(grid)
    p = a.find_path((0,0),(0,2))
    assert p is not None
    cost = path_cost(grid,p)
    # direct via (0,1) would be 101, optimal should be cheaper
    assert cost < 101
    assert cost == 6  # down around the bottom
    # path must avoid the 100 cells
    assert (0,1) not in p and (1,1) not in p

def test_no_path_exists():
    grid = [
        [1,0,1],
        [0,0,0],
        [1,0,1]
    ]
    a = AStarGrid(grid)
    p = a.find_path((0,0),(2,2))
    assert p is None

def test_start_equals_end():
    grid = [[1]]
    a = AStarGrid(grid)
    p = a.find_path((0,0),(0,0))
    assert p == [(0,0)]

def test_invalid_coordinates():
    grid = [[1,1],[1,1]]
    a = AStarGrid(grid)
    with pytest.raises(ValueError):
        a.find_path((-1,0),(1,1))
    with pytest.raises(ValueError):
        a.find_path((0,0),(2,0))

def test_start_or_end_is_wall():
    grid = [[0,1],[1,1]]
    a = AStarGrid(grid)
    assert a.find_path((0,0),(1,1)) is None
    assert a.find_path((0,1),(0,0)) is None
```

The implementation uses a min-heap for the open set, Manhattan distance as an admissible heuristic and returns the optimal minimum-cost 4-directional path.