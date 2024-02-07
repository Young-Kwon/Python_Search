"""
Description: Implement DFS to find the best path and compare its performance with the provided A* algorithm.
Function: is_valid
          heuristic
          a_star
          dfs
          print_grid
Author: Young Sang Kwon
Date: Nov 27th, 2023
Version: 1.0

This Python script is designed to find paths in a grid map from a start position to multiple goal positions. 
The script utilizes the A* search algorithm for pathfinding, considering obstacles in the grid. 
The expected output of the file includes the paths from the start position to each of the specified goal positions, 
along with the number of cells explored during the search.

Compared to the Depth-First Search (DFS) algorithm, also implemented in this script, 
A* demonstrates superior performance in terms of path optimality and the number of explored cells. 
While DFS explores paths aggressively without considering the distance to the goal, 
A* strategically chooses paths that are more likely to lead to the goal efficiently, using a heuristic function. 
As a result, A* often finds the shortest path more quickly and with fewer explored cells than DFS.
"""

import heapq
import numpy as np

# Define the grid map with obstacles (0 represents open cells, 1 represents obstacles)
grid_map = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
])

# Define the start and goal positions
start = (0, 0)
goals = [(2, 4), (4, 4), (6, 8)]  # List of goal positions

# Define possible movement directions (up, down, left, right)
directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]


def is_valid(pos, grid):
    """
    Check if a given position is valid (inside the grid and not an obstacle).

    Args:
        pos (tuple): The position to check as a tuple (x, y).
        grid (numpy.ndarray): The grid map containing obstacle information.

    Returns:
        bool: True if the position is valid, False otherwise.
    """
    x, y = pos
    return 0 <= x < grid.shape[0] and 0 <= y < grid.shape[1] and grid[x][y] == 0

def heuristic(pos, goal):
    """
    Calculate the Manhattan distance heuristic between two positions.

    Args:
        pos (tuple): The current position as a tuple (x, y).
        goal (tuple): The goal position as a tuple (x, y).

    Returns:
        int: The Manhattan distance between the two positions.
    """
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def a_star(grid, start, goal):
    """
    Find the shortest path from a start position to a goal using the A* algorithm.

    Args:
        grid (numpy.ndarray): The grid map with obstacle information.
        start (tuple): The starting position as a tuple (x, y).
        goal (tuple): The goal position as a tuple (x, y).

    Returns:
        tuple: A tuple containing the path from start to goal and a set of explored cells.
            The path is a list of positions (tuples).
    """
    open_list = [(0, start)]  # Priority queue to store nodes to be explored
    closed_set = set()  # Set to store already explored nodes
    g_score = {start: 0}  # Dictionary to store the cost from start to each node
    f_score = {start: heuristic(start, goal)}  # Dictionary to store total cost
    came_from = {}  # Dictionary to store the previous node in the path

    while open_list:
        _, current = heapq.heappop(open_list)  # Get the node with the lowest f-score

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, closed_set

        closed_set.add(current)

        for direction in directions:
            next_pos = (current[0] + direction[0], current[1] + direction[1])

            if not is_valid(next_pos, grid) or next_pos in closed_set:
                continue

            tentative_g_score = g_score[current] + 1  # Tentative cost from start to next node

            if next_pos not in [item[1] for item in open_list] or tentative_g_score < g_score[next_pos]:
                came_from[next_pos] = current
                g_score[next_pos] = tentative_g_score
                f_score[next_pos] = tentative_g_score + heuristic(next_pos, goal)
                heapq.heappush(open_list, (f_score[next_pos], next_pos))  # Add to the priority queue

    return None, closed_set

def dfs(grid, start, goal):
    """
    Performs Depth-First Search (DFS) in a grid from a start position to a goal position.
    
    Parameters:
    grid (list of lists): The grid map where 0 represents open cells and 1 represents obstacles.
    start (tuple): The starting position in the grid.
    goal (tuple): The goal position in the grid.

    Returns:
    list: The path from start to goal if exists, otherwise an empty list.
    set: The set of explored cells during the search.
    """
    # Check if start or goal is an obstacle
    if grid[start[0]][start[1]] == 1 or grid[goal[0]][goal[1]] == 1:
        return [], set()

    # Initialize the stack with the start position
    stack = [start]
    # Track the explored cells
    explored = set()
    # Store the path taken
    path = {}

    while stack:
        current = stack.pop()
        if current in explored:
            continue
        explored.add(current)

        # Check if the current position is the goal
        if current == goal:
            break

        for direction in directions:
            next_cell = (current[0] + direction[0], current[1] + direction[1])
            # Check if next cell is within grid bounds and not an obstacle
            if (0 <= next_cell[0] < len(grid) and 0 <= next_cell[1] < len(grid[0]) and 
                grid[next_cell[0]][next_cell[1]] == 0 and next_cell not in explored):
                stack.append(next_cell)
                path[next_cell] = current

    # Reconstruct path from goal to start
    dfs_path = []
    current = goal
    while current != start:
        dfs_path.append(current)
        current = path.get(current)
        if current is None:  # No path found
            return [], explored
    dfs_path.append(start)
    dfs_path.reverse()

    return dfs_path, explored

# Print the grid with A* paths, goals, path, and the path that was taken marked as "-"
def print_grid(grid_map, paths, goals):
    """
    Print the grid with A* paths, goals, and the path that was taken marked as "-".

    Args:
        grid_map (numpy.ndarray): The grid map with obstacle information.
        paths (list of lists): List of paths, where each path is a list of positions (tuples).
        goals (list of tuples): List of goal positions.

    Returns:
        None
    """
    for i in range(len(grid_map)):
        for j in range(len(grid_map[i])):
            pos = (i, j)
            if any(pos in path for path in paths):
                if pos in paths[0]:
                    print("1", end=" ")  # 1 represents the path taken by goal 1
                elif pos in paths[1]:
                    print("2", end=" ")  # 2 represents the path taken by goal 2
                elif pos in paths[2]:
                    print("3", end=" ")  # 3 represents the path taken by goal 3
            elif pos in goals:
                print("X", end=" ")  # X represents the goal
            elif grid_map[i][j] == 1:
                print("#", end=" ")  # # represents obstacles
            else:
                print(".", end=" ")  # . represents open cells
        print()  # Start a new line for the next row


# Run A* for each goal
goal_paths = []
explored_cells = []
for goal in goals:
    a_star_path, closed_set_a_star = a_star(grid_map, start, goal)
    goal_paths.append(a_star_path)
    explored_cells.append(len(closed_set_a_star))


# Print the A* paths for each goal
print("A* Paths with Goals:")
for i, (goal_path, explored) in enumerate(zip(goal_paths, explored_cells)):
    print(f"Goal {i + 1}:")
    if goal_path:
        print_grid(grid_map, goal_paths, [goals[i]])
        print(f'Path Length: {len(goal_path) - 1}')
        print(f'Explored Cells: {explored}')
    else:
        print("No path found for this goal.")
    print()

# Run dfa for each goal
goal_paths = []
explored_cells = []
for goal in goals:
    dfs_path, closed_set_dfs = dfs(grid_map, start, goal)
    goal_paths.append(dfs_path)
    explored_cells.append(len(closed_set_dfs))


# Print the dfa paths for each goal
print("dfs Paths with Goals:")
for i, (goal_path, explored) in enumerate(zip(goal_paths, explored_cells)):
    print(f"Goal {i + 1}:")
    if goal_path:
        print_grid(grid_map, goal_paths, [goals[i]])
        print(f'Path Length: {len(goal_path) - 1}')
        print(f'Explored Cells: {explored}')
    else:
        print("No path found for this goal.")
    print()    
