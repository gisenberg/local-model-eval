# Benchmark: A* Pathfinding

**Difficulty:** Medium-Hard
**Expected tests:** 6
**Skills tested:** Graph algorithms, priority queues/heaps, heuristic design, edge case handling

## Prompt

```
Implement A* pathfinding on a weighted 2D grid in Python. Requirements:

1. Class AStarGrid with __init__(self, grid: List[List[int]]) where grid values represent movement cost (0 = impassable wall, positive int = cost to enter that cell)
2. find_path(start: Tuple[int, int], end: Tuple[int, int]) -> Optional[List[Tuple[int, int]]] — return shortest path as list of (row, col) coordinates from start to end inclusive, or None if no path exists
3. Support 4-directional movement (up, down, left, right) — no diagonals
4. Use Manhattan distance as the heuristic
5. Handle edge cases: start == end (return [start]), start or end is a wall (return None), start or end out of bounds (raise ValueError)
6. The path must be optimal (minimum total cost)
7. Use a min-heap (heapq) for the open set
8. Include type hints throughout and a brief docstring on each method
9. Write 6 pytest tests covering: simple path on uniform grid, path around obstacles, weighted grid (path prefers lower-cost cells), no path exists (fully blocked), start equals end, and invalid coordinates. Assert both path validity and optimality (total cost).
```

## What Makes This a Good Benchmark

- Requires correct A* implementation (not just BFS/Dijkstra)
- Manhattan distance heuristic must be admissible for optimality
- Weighted grids test that the algorithm considers costs, not just distances
- Multiple valid optimal paths exist — tests must account for this
- Edge cases (walls, bounds, self-loop) test defensive programming

## Common Failure Modes Observed

1. **Hardcoded path assertions** (Nemotron 30B): Tests assert a specific path when multiple optimal routes exist
2. **Wrong cost arithmetic** (Qwen 9B): Test expects `total_cost == 8` but correct answer is 5
3. **Incorrect ValueError expectations** (Nemotron 30B): Tests expect ValueError for valid coordinates

## Evaluation Criteria

- Does the code import without errors?
- Do all 6 tests pass when run with `pytest -v`?
- Tests should validate path optimality (total cost), not specific path sequences
- No manual fixes allowed
