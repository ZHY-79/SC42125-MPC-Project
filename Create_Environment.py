"""
This script implements functions to easily create a environment including walls, static obstacles and dynamic obstacles.
Two typical scenarios are included (A open environment and a parking lot).
Also, several functions are created fot visualize the environment and movement of the robot.

The main features include:

1. **Generate_Walls**:
   - Generate wall boundary for the environment.
2. **Generate_Obstacle_Centers**:
   - Generate n random obstacle centers inside the wall boundary.
3. **Generate_Obstacles**:
   - Generate urdfenvs static obstacles according to obstacle centers from **Generate_Obstacle_Centers**.
4. **Generate_Dynamic_Obstacles**:
   - Generate urdfenvs dynamic obstacles.
5. **create_grid_map**:
    - Create grid map with certain resolution according to the environment.
6. **Environment Visualization**:
   - Draws the obstacles, walls and vehicles on a grid map using Matplotlib.

This script is modular and can be integrated into larger systems for simulation environment set-up and visualization tasks.
"""
from urdfenvs.scene_examples.obstacles import *
from mpscenes.obstacles import *
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon


def Generate_Walls(length, width, Type="Square", print_log = True):
    """
    Generate wall boundary for the environment.

    Parameters:
    - length: Length of the enclosed area.
    - width: Width of the enclosed area.
    - Type: Choose from open environment (Square) or parking lot (Parking Lot).
    - print_log: Print debug information.

    Returns:
    - Walls: A List of BoxObstacle type objects containing wall boundaries
    - Dicts: A dictionary containing some detailed information about walls
    """
    
    # Create walls depending on Type
    if Type == "Square":
        
        # Create wall information
        Wall1_dict = {
            'type': 'box',
            'geometry': {
                'position': [0, width / 2, 0.75],
                'width': 0.2,
                'height': 1.5,
                'length': length,
            },
            'movable': False,
        }

        Wall2_dict = {
            'type': 'box',
            'geometry': {
                'position': [0, -width / 2, 0.75],
                'width': 0.2,
                'height': 1.5,
                'length': length,
            },
            'movable': False,
        }

        Wall3_dict = {
            'type': 'box',
            'geometry': {
                'position': [length / 2, 0, 0.75],
                'width': width,
                'height': 1.5,
                'length': 0.2,
            },
            'movable': False,
        }

        Wall4_dict = {
            'type': 'box',
            'geometry': {
                'position': [-length / 2, 0, 0.75],
                'width': width,
                'height': 1.5,
                'length': 0.2,
            },
            'movable': False,
        }
        
        # Create BoxObstacles according to dictionaries
        Wall1 = BoxObstacle(name="Wall1", content_dict=Wall1_dict)
        Wall2 = BoxObstacle(name="Wall2", content_dict=Wall2_dict)
        Wall3 = BoxObstacle(name="Wall3", content_dict=Wall3_dict)
        Wall4 = BoxObstacle(name="Wall4", content_dict=Wall4_dict)

        Walls = [Wall1, Wall2, Wall3, Wall4]
        Dicts = [Wall1_dict, Wall2_dict, Wall3_dict, Wall4_dict]
    
    # Another environmnet type
    elif Type == "Parking Lot":
        
        # Print debug information
        if print_log:
            print("!!! Width and Length will be fixed !!!")
            
    # Size of the environment will be fixed in the parking lot scenario
    length = 50
    width = 80
    
    # A dictionary containing all the information
    walls_properties = [
        {"name": "Wall1", "position": [0, width / 2, 0.75], "dimensions": [0.2, 1.5, length]},
        {"name": "Wall2", "position": [0, -width / 2, 0.75], "dimensions": [0.2, 1.5, length]},
        {"name": "Wall3", "position": [length / 2, 0, 0.75], "dimensions": [width, 1.5, 0.2]},
        {"name": "Wall4", "position": [-length / 2, 0, 0.75], "dimensions": [width, 1.5, 0.2]},
        {"name": "Wall5", "position": [15, 17.5, 0.75], "dimensions": [25, 1.5, 0.2]},
        {"name": "Wall6", "position": [15, -17.5, 0.75], "dimensions": [25, 1.5, 0.2]},
        {"name": "Wall7", "position": [5, 17.5, 0.75], "dimensions": [25, 1.5, 0.2]},
        {"name": "Wall8", "position": [5, -17.5, 0.75], "dimensions": [25, 1.5, 0.2]},
        {"name": "Wall9", "position": [-15, 17.5, 0.75], "dimensions": [25, 1.5, 0.2]},
        {"name": "Wall10", "position": [-15, -17.5, 0.75], "dimensions": [25, 1.5, 0.2]},
        {"name": "Wall11", "position": [-5, 17.5, 0.75], "dimensions": [25, 1.5, 0.2]},
        {"name": "Wall12", "position": [-5, -17.5, 0.75], "dimensions": [25, 1.5, 0.2]},
        {"name": "Wall13", "position": [10, 30, 0.75], "dimensions": [0.2, 1.5, 10]},
        {"name": "Wall14", "position": [10, -30, 0.75], "dimensions": [0.2, 1.5, 10]},
        {"name": "Wall15", "position": [10, 5, 0.75], "dimensions": [0.2, 1.5, 10]},
        {"name": "Wall16", "position": [10, -5, 0.75], "dimensions": [0.2, 1.5, 10]},
        {"name": "Wall17", "position": [-10, 30, 0.75], "dimensions": [0.2, 1.5, 10]},
        {"name": "Wall18", "position": [-10, -30, 0.75], "dimensions": [0.2, 1.5, 10]},
        {"name": "Wall19", "position": [-10, 5, 0.75], "dimensions": [0.2, 1.5, 10]},
        {"name": "Wall20", "position": [-10, -5, 0.75], "dimensions": [0.2, 1.5, 10]},
    ]
    
    # Create BoxObstacles iteratively
    Walls = []
    Dicts = []
    for wall in walls_properties:
        wall_dict = {
            'type': 'box',
            'geometry': {
                'position': wall["position"],
                'width': wall["dimensions"][0],
                'height': wall["dimensions"][1],
                'length': wall["dimensions"][2],
            },
            'movable': False,
        }
        Walls.append(BoxObstacle(name=wall["name"], content_dict=wall_dict))
        Dicts.append(wall_dict)
    
    return Walls, Dicts


def Generate_Obstacle_Centers(length, width, n_obstacles, min_distance=5.0, Type="Square"):
    """
    Generate n random obstacle centers inside the wall boundary, avoiding corners and ensuring
    obstacles are at least `min_distance` apart.

    Parameters:
    - length: Length of the enclosed area.
    - width: Width of the enclosed area.
    - n_obstacles: Number of random obstacle centers to generate.
    - min_distance: Minimum distance between obstacles.

    Returns:
    - obstacles: List of dictionaries with 'center' (coordinates) and 'type' (box or sphere).
    """
    
    if Type == "Square":
        centers = []
        # Margin to keep obstacles away from the walls
        margin = 1.5  
        offset = 8
        corner_exclusion = [  # Define corner regions to exclude
            [-length / 2, -width / 2, -length / 2 + offset, -width / 2 + offset],  # Bottom-left
            [length / 2 - offset, -width / 2, length / 2, -width / 2 + offset],  # Bottom-right
            [-length / 2, width / 2 - offset, -length / 2 + offset, width / 2],  # Top-left
            [length / 2 - offset, width / 2 - offset, length / 2, width / 2],  # Top-right
        ]

        def is_in_corner(x, y):
            # Check if a point (x, y) is in any corner exclusion zone.
            for x_min, y_min, x_max, y_max in corner_exclusion:
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    return True
            return False

        def distance(p1, p2):
            # Calculate Euclidean distance between two points (x1, y1) and (x2, y2).
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        obstacles = []
        while len(obstacles) < n_obstacles:
            x = random.uniform(-length / 2 + margin, length / 2 - margin)
            y = random.uniform(-width / 2 + margin, width / 2 - margin)

            # Skip if the point is in a corner region
            if is_in_corner(x, y):
                continue

            # Check if the new obstacle is too close to existing ones
            too_close = False
            for obstacle in obstacles:
                if distance([x, y], obstacle['center']) < min_distance:
                    too_close = True
                    break

            if too_close:
                continue

            # Assign a random type (box or sphere) to the obstacle
            obstacle_type = random.choice(['box', 'sphere'])

            # Add the obstacle with its center and type
            obstacles.append({'center': [x, y], 'type': obstacle_type, 'direction': None})
            
    # Parking lot scenario
    elif Type == "Parking Lot":
        # Define obstacles
        all_obstacles = [
            {'center': [-23, -22], 'type': 'box', 'direction': 'Along_width'},
            {'center': [-16.5, -10], 'type': 'box', 'direction': 'Along_width'},
            {'center': [-23, 10], 'type': 'box', 'direction': 'Along_width'},
            {'center': [-16.5, 25], 'type': 'box', 'direction': 'Along_width'},
            {'center': [-3.5, 12], 'type': 'box', 'direction': 'Along_width'},
            {'center': [-11, 31.5], 'type': 'box', 'direction': 'Along_length'},
            {'center': [-3.5, -12], 'type': 'box', 'direction': 'Along_width'},
            {'center': [-8, -3.5], 'type': 'box', 'direction': 'Along_length'},
            {'center': [3.5, -25], 'type': 'box', 'direction': 'Along_width'},
            {'center': [-10, -38.5], 'type': 'box', 'direction': 'Along_length'},
            {'center': [11, 32], 'type': 'box', 'direction': 'Along_length'},
            {'center': [8, 3.5], 'type': 'box', 'direction': 'Along_length'},
            {'center': [9, -31.5], 'type': 'box', 'direction': 'Along_length'},
            {'center': [16.5, 12], 'type': 'box', 'direction': 'Along_width'},
            {'center': [16.5, -15], 'type': 'box', 'direction': 'Along_width'},
            {'center': [23.5, -30], 'type': 'box', 'direction': 'Along_width'},
        ]
        obstacles = []
        
        # Randomly select n obstacles
        selected_obstacles = random.sample(all_obstacles, n_obstacles)
        obstacles.extend(selected_obstacles)
        
    return obstacles


def Generate_Obstacles(obstacles, Type="Square"):
    """
    Generate static obstacles for urdfenvs

    Parameters:
    - obstacles: A dictionary of the center of obstacles generated from Generate_Obstacle_Centers
    - Type: Choose from open environment (Square) or parking lot (Parking Lot).

    Returns:
    - Obstacles: A List of BoxObstacle/SphereObstacle type objects containing all the generated obstacles
    - Dicts: A dictionary containing some detailed information about obstacles
    """
    
    Obstacles = []
    Dicts = []
    # Creating obstacles iteratively
    for i, obstacle in enumerate(obstacles):
        obstacle_name = f"obstacle_{i + 1}"
        
        # First create BoxObstacle
        if obstacle['type'] == 'box':
            
            # Judging scenario type
            if Type == "Square":
                # Randomly select direction of the obstacle
                direction = random.choice(['Along_length', 'Along_width'])
                if direction == 'Along_width':
                    # Randomly generate the size of the obstacle
                    height = random.uniform(0.5, 2)
                    Box_dict = {
                        'type': 'box',
                        'geometry': {
                            'position': [obstacle['center'][0], obstacle['center'][1], height / 2],
                            'width': random.uniform(0.5, 2),
                            'height': height,
                            'length': random.uniform(0.5, 2),
                        },
                        'movable': False,
                    }
                    Dicts.append(Box_dict)
                    Obstacles.append(BoxObstacle(name=obstacle_name, content_dict=Box_dict))
                else:
                    # Randomly generate the size
                    height = random.uniform(0.5, 2)
                    Box_dict = {
                        'type': 'box',
                        'geometry': {
                            'position': [obstacle['center'][0], obstacle['center'][1], height / 2],
                            'width': random.uniform(0.5, 2),
                            'height': height,
                            'length': random.uniform(0.5, 2),
                        },
                        'movable': False,
                    }
                    Dicts.append(Box_dict)
                    Obstacles.append(BoxObstacle(name=obstacle_name, content_dict=Box_dict))
            # Judging scenario type
            elif Type == "Parking Lot":
                # Obstacles are of same height in the parking lot scenario
                height = 1.2
                # Judging the direction
                if obstacle['direction'] == 'Along_width':
                    Box_dict = {
                        'type': 'box',
                        'geometry': {
                            'position': [obstacle['center'][0], obstacle['center'][1], height / 2],
                            # Randomly generate width and length
                            'width': random.uniform(4.2, 4.8),
                            'height': height,
                            'length': random.uniform(1.8, 2.0),
                        },
                        'movable': False,
                    }
                    Dicts.append(Box_dict)
                    Obstacles.append(BoxObstacle(name=obstacle_name, content_dict=Box_dict))
                else:
                    Box_dict = {
                        'type': 'box',
                        'geometry': {
                            'position': [obstacle['center'][0], obstacle['center'][1], height / 2],
                            # Randomly generate width and length
                            'width': random.uniform(1.8, 2.0),
                            'height': height,
                            'length': random.uniform(4.2, 4.8),
                        },
                        'movable': False,
                    }
                    Dicts.append(Box_dict)
                    Obstacles.append(BoxObstacle(name=obstacle_name, content_dict=Box_dict))
                    
        # Second create  SphereObstacle
        if obstacle['type'] == 'sphere':
            # Randomly generate radius
            radius = random.uniform(0.5, 1)
            Sphere_dict = {
                "type": "sphere",
                "geometry": {
                    "position": [obstacle['center'][0], obstacle['center'][1], radius],
                    "radius": radius,
                },
            }
            Dicts.append(Sphere_dict)
            Obstacles.append(SphereObstacle(name=obstacle_name, content_dict=Sphere_dict))

    return Obstacles, Dicts


def Generate_Dynamic_Obstacles(n_steps, n_obstacles=12, dt=0.01, Type="Parking Lot"):
    """
    Generate dynamic obstacles for urdfenvs

    Parameters:
    - n_steps: Total steps of the simulation
    - n_obstacles: Number of dynamic obstacles
    - dt: Duration of one simulation step
    - Type: Choose from open environment (Square) or parking lot (Parking Lot).

    Returns:
    - Dynamic_Obstacles: A List of DynamicSphereObstacle type objects
    - Dicts: A dictionary containing some detailed information about dynamic obstacles
    """
    Dicts = []
    Dynamic_Obstacles = []
    # Judging scenario type
    if Type == "Parking Lot":
        # Crate time steps
        t_values = np.linspace(0, n_steps * dt, num=n_steps)

        # Trajectory configuration for Obstalces
        obstacle_configs = [
            (4 * np.sin(0.5 * t_values + np.pi / 6), -5 * np.ones_like(t_values), 0.3 * np.ones_like(t_values)),
            (4 * np.sin(0.5 * t_values), 5 * np.ones_like(t_values), 0.3 * np.ones_like(t_values)),
            #(4.5 * np.ones_like(t_values), 4 * np.sin(0.5 * t_values + np.pi / 6), 0.3 * np.ones_like(t_values)),
            #(-4.5 * np.ones_like(t_values), 4 * np.sin(0.5 * t_values + np.pi / 4), 0.3 * np.ones_like(t_values)),
            #(4 * np.sin(0.5 * t_values) - 20, 5 * np.ones_like(t_values), 0.3 * np.ones_like(t_values)),
            #(4 * np.sin(0.5 * t_values + np.pi / 6) - 20, -5 * np.ones_like(t_values), 0.3 * np.ones_like(t_values)),
            #(4 * np.sin(0.5 * t_values) + 20, 5 * np.ones_like(t_values), 0.3 * np.ones_like(t_values)),
            #(4 * np.sin(0.5 * t_values + np.pi / 6) + 20, -5 * np.ones_like(t_values), 0.3 * np.ones_like(t_values)),
            #(4.5 * np.ones_like(t_values), 4 * np.sin(0.5 * t_values) + 35, 0.3 * np.ones_like(t_values)),
            #(-4.5 * np.ones_like(t_values), 4 * np.sin(0.5 * t_values + np.pi / 4) + 35, 0.3 * np.ones_like(t_values)),
            (4.5 * np.ones_like(t_values), 4 * np.sin(0.5 * t_values) - 35, 0.3 * np.ones_like(t_values)),
            (-8 * np.ones_like(t_values), 4 * np.sin(0.5 * t_values) - 35, 0.3 * np.ones_like(t_values)),
        ]
        
        # Creating all the dynamic obstacles iteratively
        for i, (x_traj, y_traj, z_traj) in enumerate(obstacle_configs):
            config_dict = {
                "type": "sphere",
                "geometry": {
                    "trajectory": {
                        "controlPoints": np.vstack([x_traj, y_traj, z_traj]).T.tolist(),
                        "degree": 3,
                        "duration": n_steps * dt,
                    },
                    "radius": 0.3,
                },
            }

            obstacle = DynamicSphereObstacle(
                name=f"dynamic_obstacle_{i + 1}", content_dict=config_dict
            )
            Dynamic_Obstacles.append(obstacle)
            Dicts.append(config_dict)

    return Dynamic_Obstacles, Dicts


def Generate_GoalPos(i):
    """
    Generate Goal position

    Parameters:
    - i: Index of goal point

    Returns:
    - Goal: Goal position (np.array([x, y, theta]))
    - Goal_index: Index of goal point
    """
    
    # All goals containing intermediate points
    All_Goals = [
        np.array([[-20, 20, np.pi / 2]]),
        np.array([
            [0, 0, np.pi / 2],
            [0, 35, np.pi],
            [-20, 20, -np.pi / 2]]),
        np.array([
            [-20, 35, np.pi / 2],
            [0, 23, -np.pi / 2]]),
        np.array([[0, 23, np.pi / 2]]),
        np.array([[20, 25, np.pi / 2]]),
        np.array([
            [-20, 35, np.pi / 2],
            [20, 25, -np.pi / 2]]),
        np.array([
            [-20, 0, 0],
            [0, -15, -np.pi / 2]]),
        np.array([
            [0, -35, np.pi / 2],
            [0, -15, np.pi / 2]]),
        np.array([
            [-20, 0, 0],
            [20, -17, -np.pi / 2]]),
        np.array([
            [0, -35, np.pi / 2],
            [20, -17, np.pi / 2]]),
    ]
    
    '''!!!Randomly choose goal position is not used!!!'''
    # Goal = np.random.choice(All_Goals)
    # Goal = All_Goals[1]
    # Goal_index = np.random.choice(len(All_Goals))  # 1 2 4 5 8
    Goal_index = i
    Goal = All_Goals[Goal_index]  

    return Goal, Goal_index


def create_grid_map(wall_dicts, obstacle_dicts, grid_size=0.1):
    """
    Generate a grid map based on walls and obstacles.

    Parameters:
    wall_dicts -- List of dictionaries representing walls.
    obstacle_dicts -- List of dictionaries representing obstacles.
    grid_size -- Resolution of the grid map in meters (default: 0.1).

    Returns:
    grid_map -- NumPy array representing the grid map (1: occupied, 0: free).
    min_x, max_x -- Minimum and maximum x-coordinates of the map.
    min_y, max_y -- Minimum and maximum y-coordinates of the map.
    """
    # Initialize boundaries
    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    # Update map boundaries based on objects
    def update_bounds(obj):
        position = obj['geometry']['position']
        if obj['type'] == 'box':
            length = obj['geometry']['length']
            width = obj['geometry']['width']
            nonlocal min_x, max_x, min_y, max_y
            min_x = min(min_x, position[0] - length / 2)
            max_x = max(max_x, position[0] + length / 2)
            min_y = min(min_y, position[1] - width / 2)
            max_y = max(max_y, position[1] + width / 2)
        elif obj['type'] == 'sphere':
            radius = obj['geometry']['radius']
            min_x = min(min_x, position[0] - radius)
            max_x = max(max_x, position[0] + radius)
            min_y = min(min_y, position[1] - radius)
            max_y = max(max_y, position[1] + radius)

    # Update boundaries for walls and obstacles
    for wall in wall_dicts:
        update_bounds(wall)

    for obstacle in obstacle_dicts:
        update_bounds(obstacle)

    # Calculate grid map dimensions
    grid_width = int((max_x - min_x) / grid_size) + 1
    grid_height = int((max_y - min_y) / grid_size) + 1

    # Initialize grid map
    grid_map = np.zeros((grid_height, grid_width), dtype=np.int8)

    # Fill grid map with walls and obstacles
    def fill_grid(obj, value):
        position = obj['geometry']['position']
        if obj['type'] == 'box':
            length = obj['geometry']['length']
            width = obj['geometry']['width']
            x_min = int((position[0] - length / 2 - min_x) / grid_size)
            x_max = int((position[0] + length / 2 - min_x) / grid_size)
            y_min = int((position[1] - width / 2 - min_y) / grid_size)
            y_max = int((position[1] + width / 2 - min_y) / grid_size)
            grid_map[y_min:y_max + 1, x_min:x_max + 1] = value
        elif obj['type'] == 'sphere':
            radius = obj['geometry']['radius']
            cx = int((position[0] - min_x) / grid_size)
            cy = int((position[1] - min_y) / grid_size)
            r = int(radius / grid_size)
            for y in range(max(0, cy - r), min(grid_height, cy + r + 1)):
                for x in range(max(0, cx - r), min(grid_width, cx + r + 1)):
                    if (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2:
                        grid_map[y, x] = value

    # Populate the grid map
    for wall in wall_dicts:
        fill_grid(wall, 1)

    for obstacle in obstacle_dicts:
        fill_grid(obstacle, 1)

    return grid_map, min_x, max_x, min_y, max_y


def update_vehicle(ax, positions):
    """
    Update the vehicle's position and orientation on the map.

    Parameters:
    ax -- Matplotlib Axes object for plotting.
    positions -- List or array [x, y, heading] representing the vehicle's position
                 and orientation in world coordinates.
    """
    position = positions[:2]
    heading = positions[2]
    front_dist = 2.34 + 1.45  # Distance from CoG to the front
    rear_dist = 2.42 - 1.45  # Distance from CoG to the rear
    side_dist = 1.05  # Vehicle's half-width

    # Calculate the corners of the vehicle in the world frame
    cx, cy = position
    corners = np.array([
        [front_dist, side_dist],
        [front_dist, -side_dist],
        [-rear_dist, -side_dist],
        [-rear_dist, side_dist]
    ])

    # Apply rotation to the corners
    rotation = np.array([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)]
    ])
    rotated_corners = (rotation @ corners.T).T + [cx, cy]

    # Draw the vehicle polygon
    vehicle_polygon = Polygon(rotated_corners, closed=True, edgecolor='red', facecolor='none', linewidth=2, alpha=0.5)
    ax.add_patch(vehicle_polygon)

    # Draw the heading arrow
    head_x = cx + 1.5 * np.cos(heading)
    head_y = cy + 1.5 * np.sin(heading)
    arrow = plt.Arrow(cx, cy, head_x - cx, head_y - cy, color='blue', width=0.5)
    ax.add_artist(arrow)

    # Draw the center of gravity
    ax.scatter([cx], [cy], color='blue', zorder=5)

    # Refresh the plot
    plt.draw()
    plt.pause(1)


def initialize_plot(grid_map, min_x, max_x, min_y, max_y):
    """
    Initialize the plot for visualizing the grid map and the vehicle.

    Parameters:
    grid_map -- NumPy array representing the grid map.
    min_x, max_x -- Minimum and maximum x-coordinates of the map.
    min_y, max_y -- Minimum and maximum y-coordinates of the map.

    Returns:
    fig, ax -- Matplotlib figure and axes for plotting.
    """
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(grid_map, origin="lower", cmap="Greys", extent=(min_x, max_x, min_y, max_y))

    ax.set_title("Grid Map with Vehicle")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    return fig, ax

from matplotlib.patches import Circle
previous_scatter = None
obstacle_circles = []

def update_vehicle_dynobs(ax, positions, dynamic_obstacles, x, y):
    """
    Update the vehicle's position and orientation on the map. Also, update the planned trajectory from MPC on the map.

    Parameters:
    ax -- Matplotlib Axes object for plotting.
    positions -- List or array [x, y, heading] representing the vehicle's position
                 and orientation in world coordinates.
    x,y -- Trajectory points generated from the result of MPC.
    """
    
    # Clear dynamic obstacles from last step
    global previous_scatter, obstacle_circles
    for circle in obstacle_circles:
        circle.remove()
    obstacle_circles = []
    
    # Clear vehicle from last step
    #for patch in ax.patches:
        #patch.remove()
    #for artist in ax.artists:
        #artist.remove()

    # Vehicle position and orientation
    position = positions[:2]
    heading = positions[2]
    
    # Geometry information anout the vehicle
    front_dist = 2.34 + 1.45
    rear_dist = 2.42 - 1.45
    side_dist = 1.05

    # Calculate the corners of the vehicle in the world frame
    cx, cy = position
    corners = np.array([
        [front_dist, side_dist],  # Front Right
        [front_dist, -side_dist],  # Front Left
        [-rear_dist, -side_dist],  # Rear Left
        [-rear_dist, side_dist]  # Rear Right
    ])

    # Rotation matrix
    rotation = np.array([
        [np.cos(heading), -np.sin(heading)],
        [np.sin(heading), np.cos(heading)]
    ])
    rotated_corners = (rotation @ corners.T).T + [cx, cy]

    # Draw the vehicle 
    vehicle_polygon = Polygon(rotated_corners, closed=True, edgecolor='red', facecolor='none', linewidth=2, alpha=0.5)
    ax.add_patch(vehicle_polygon)

    # Draw the direction of the vehicle 
    head_x = cx + 1.5 * np.cos(heading)
    head_y = cy + 1.5 * np.sin(heading)
    arrow = plt.Arrow(cx, cy, head_x - cx, head_y - cy, color='blue', width=0.5)
    ax.add_artist(arrow)

    # Draw the reference point of the vehicle (center of the rear axle)
    ax.scatter([cx], [cy], color='blue', zorder=5)

    # Draw dynamic objects
    for obstacle in dynamic_obstacles:
        obs_x, obs_y = obstacle
        obstacle_circle = Circle((obs_x, obs_y), 0.3, color='orange', alpha=0.5)
        ax.add_patch(obstacle_circle)
        obstacle_circles.append(obstacle_circle)
        
    # Remove trajectory from last step
    if previous_scatter is not None:
        previous_scatter.remove()
    # Plot Trajectory
    previous_scatter = ax.scatter(x, y, color='purple', s=4, zorder=5)
    # Refreash the plot
    plt.draw()
    plt.pause(0.1)
