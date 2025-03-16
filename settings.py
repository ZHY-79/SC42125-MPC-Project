import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, Polygon
def CoG_to_RearAxle(CoG_pos, dis=1.45):
    x_cog = CoG_pos[0]
    y_cog = CoG_pos[1]
    theta_cog = CoG_pos[2]
    
    x_r = x_cog - dis * np.cos(theta_cog)
    y_r = y_cog - dis * np.sin(theta_cog)
    
    return np.array([x_r, y_r, theta_cog])

def force_axis_limits(ax):
    """强制坐标轴范围为-50到50"""
    # 设置坐标轴范围
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    
    # 禁用自动缩放
    ax.set_autoscale_on(False)
    
    # 显式设置刻度位置
    ticks = [-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    
    # 刷新图形
    ax.figure.canvas.draw()

def initialize_plot():
    # 使用更大的图形大小，并设置小边距
    plt.rcParams.update({'figure.autolayout': True})
    fig = plt.figure(figsize=(12, 10), tight_layout=True)
    ax = fig.add_subplot(111)
    
    # 强制设置坐标轴范围
    force_axis_limits(ax)
    
    # 设置网格和标签
    ax.grid(True)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Vehicle Trajectory Tracking with MPC')
    
    # 添加坐标轴范围提示
    ax.text(48, 48, '50x50', fontsize=12, ha='right', va='top', 
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    return fig, ax

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

    # Clean up previous vehicle drawings
    for patch in ax.patches:
        if isinstance(patch, Polygon) and patch.get_edgecolor()[0:3] == (1, 0, 0):  # Red polygons
            patch.remove()

    # Draw the vehicle polygon
    vehicle_polygon = Polygon(rotated_corners, closed=True, edgecolor='red', facecolor='blue', 
                            alpha=0.3, linewidth=2)
    ax.add_patch(vehicle_polygon)

    # Draw the heading arrow
    head_x = cx + 1.5 * np.cos(heading)
    head_y = cy + 1.5 * np.sin(heading)
    
    # Clean up previous arrows
    for artist in ax.get_children():
        if isinstance(artist, plt.Arrow):
            artist.remove()
            
    arrow = plt.Arrow(cx, cy, head_x - cx, head_y - cy, color='black', width=0.5)
    ax.add_artist(arrow)

    # Force coordinate range to stay fixed
    force_axis_limits(ax)
    
    # Pause briefly to update display
    plt.pause(0.001)