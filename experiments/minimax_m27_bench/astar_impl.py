grid = [
    [1, 3, 1, 1, 1],
    [1, 1, 1, 3, 1],
    [1, 3, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1]
]

astar = AStarGrid(grid)
path = astar.find_path((0, 0), (4, 4))

print(f"Path: {path}")
print(f"Total cost: {sum(grid[r][c] for r, c in path)}")
