"""
Tests for AStarGrid pathfinding implementation.
"""

import pytest
from astar_grid import AStarGrid


class TestAStarGrid:
    """Test suite for A* pathfinding on a weighted 2D grid."""

    def test_simple_path_uniform_grid(self):
        """Test simple path on a uniform grid with no obstacles."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        # Manhattan distance on a uniform grid = 4 steps
        assert len(path) == 5
        # Total cost should be 4 (each step cost 1)
        total_cost = sum(grid[r][c] for r, c in path[1:])
        assert total_cost == 4

    def test_path_around_obstacles(self):
        """Test path that must go around walls."""
        grid = [
            [1, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 3))
        
        assert path is not None
        assert path[0] == (0, 0)
        assert path[-1] == (2, 3)
        # Path should go down, right, right, down, down
        # or similar valid route around the wall
        assert (1, 0) in path or (0, 1) in path

    def test_weighted_grid_prefers_lower_cost(self):
        """Test that the path prefers lower-cost cells over higher-cost ones."""
        grid = [
            [1, 10, 1],
            [1, 1,  1],
            [1, 10, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (0, 2))
        
        assert path is not None
        # Should go through the middle row to avoid high-cost cells
        assert (1, 1) in path
        # Total cost should be 3 (1+1+1) going through middle
        # rather than 11+1+11 = 23 going through top/bottom
        total_cost = sum(grid[r][c] for r, c in path)
        assert total_cost == 3

    def test_no_path_exists(self):
        """Test when there is no path due to complete blockage."""
        grid = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((0, 0), (2, 2))
        
        assert path is None

    def test_start_equals_end(self):
        """Test when start and end are the same cell."""
        grid = [
            [1, 1],
            [1, 1]
        ]
        astar = AStarGrid(grid)
        path = astar.find_path((1, 1), (1, 1))
        
        assert path is not None
        assert path == [(1, 1)]

    def test_invalid_coordinates(self):
        """Test that ValueError is raised for out-of-bounds coordinates."""
        grid = [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar = AStarGrid(grid)
        
        # Out of bounds
        with pytest.raises(ValueError):
            astar.find_path((0, 0), (3, 3))
        
        with pytest.raises(ValueError):
            astar.find_path((-1, 0), (2, 2))
        
        # Start is a wall
        grid_with_wall = [
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ]
        astar_wall = AStarGrid(grid_with_wall)
        path = astar_wall.find_path((0, 0), (2, 2))
        assert path is None
        
        # End is a wall
        path = astar_wall.find_path((0, 1), (0, 0))
        assert path is None
