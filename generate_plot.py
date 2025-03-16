import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.interpolate import splprep, splev
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D

class BicycleModel:
    def __init__(self, x=0, y=0, theta=0, L=4.76, front_dist=3.79, rear_dist=0.97, half_width=1.05):
        """
        Initialize bicycle model
        x, y: position coordinates
        theta: vehicle heading angle (radians)
        L: wheelbase
        front_dist: distance from CoG to front axle
        rear_dist: distance from CoG to rear axle
        half_width: vehicle's half-width
        """
        self.x = x
        self.y = y
        self.theta = theta
        self.L = L
        self.front_dist = front_dist
        self.rear_dist = rear_dist
        self.half_width = half_width
    
    def update(self, v, phi, dt):
        """
        Update position based on bicycle kinematic model
        v: velocity
        phi: front wheel steering angle (radians)
        dt: time step
        """
        # Bicycle kinematic equations
        self.x = self.x + v * np.cos(self.theta) * dt
        self.y = self.y + v * np.sin(self.theta) * dt
        self.theta = self.theta + (v / self.L) * np.tan(phi) * dt
        
        return self.x, self.y, self.theta

def draw_trajectory():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Generate Bicycle-Constrained Reference Trajectory")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.grid(True)
    
    # Define start and end points
    fixed_start = (-46, 46)
    fixed_end = (46, -46)
    
    ax.plot(*fixed_start, 'go', markersize=8, label='Start')
    ax.plot(*fixed_end, 'ro', markersize=8, label='End')
    
    # Collect control points from user clicks
    control_points = []
    
    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            control_points.append((event.xdata, event.ydata))
            
            # Draw control point
            ax.plot(event.xdata, event.ydata, 'ko', markersize=6)
            
            # If multiple control points, draw temporary connecting lines
            if len(control_points) > 1:
                x_vals = [p[0] for p in control_points]
                y_vals = [p[1] for p in control_points]
                
                # Clear previous temporary lines
                for line in ax.lines:
                    if line.get_linestyle() == '--' and line.get_color() == 'k':
                        line.remove()
                
                ax.plot(x_vals, y_vals, 'k--', alpha=0.5)
            
            fig.canvas.draw()
    
    def on_key(event):
        if event.key == 'enter' and len(control_points) > 0:
            # Add start and end points
            all_points = [fixed_start] + control_points + [fixed_end]
            
            # Generate trajectory satisfying bicycle constraints
            generate_bicycle_trajectory(ax, all_points)
            
            # Add legend
            ax.legend()
            fig.canvas.draw()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()

def generate_bicycle_trajectory(ax, control_points):
    """
    Generate a reference trajectory satisfying bicycle constraints based on control points
    Method: First generate a smooth path using splines, then optimize curvature to ensure constraints
    """
    # Use spline interpolation to generate initial reference path
    cp_array = np.array(control_points)
    tck, u = splprep([cp_array[:,0], cp_array[:,1]], s=50.0, k=min(3, len(cp_array)-1))
    u_fine = np.linspace(0, 1, 1000)
    x_smooth, y_smooth = splev(u_fine, tck)
    
    # Plot initial smooth path
    ax.plot(x_smooth, y_smooth, 'g--', linewidth=1.5, alpha=0.7, label='Initial Smooth Path')
    
    # Calculate path curvature
    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    
    # Curvature calculation: κ = |x'y'' - y'x''|/(x'^2 + y'^2)^(3/2)
    curvature = np.abs(dx*ddy - dy*ddx) / (dx**2 + dy**2)**(1.5)
    
    # Bicycle model parameters (based on provided vehicle data)
    front_dist = 2.34 + 1.45  # Distance from CoG to the front
    rear_dist = 2.42 - 1.45   # Distance from CoG to the rear
    wheelbase = front_dist + rear_dist  # Wheelbase = 4.76m
    vehicle_half_width = 1.05  # Vehicle's half-width
    max_steering_angle = np.pi/6  # Maximum steering angle (30 degrees)
    
    # Maximum allowed curvature
    max_curvature = np.tan(max_steering_angle) / wheelbase
    
    # Find points exceeding maximum curvature
    excessive_curve_points = np.where(curvature > max_curvature)[0]
    if len(excessive_curve_points) > 0:
        print(f"Detected {len(excessive_curve_points)} points with curvature exceeding bicycle motion constraints")
        
        # Mark points with excessive curvature on the plot
        for idx in excessive_curve_points:
            ax.plot(x_smooth[idx], y_smooth[idx], 'rx', markersize=4)
    
    # Generate trajectory based on bicycle model
    # Initialize bicycle at starting point
    start_x, start_y = control_points[0]
    
    # Determine initial heading
    if len(control_points) > 1:
        next_x, next_y = control_points[1]
        initial_theta = np.arctan2(next_y - start_y, next_x - start_x)
    else:
        # If only start and end points, use the direct line direction
        end_x, end_y = control_points[-1]
        initial_theta = np.arctan2(end_y - start_y, end_x - start_x)
    
    # Create bicycle model (with actual vehicle parameters)
    bicycle = BicycleModel(start_x, start_y, initial_theta, wheelbase, front_dist, rear_dist, vehicle_half_width)
    
    # Generate trajectory that satisfies bicycle constraints
    trajectory = [(bicycle.x, bicycle.y, bicycle.theta)]
    
    # Trajectory generation parameters
    dt = 0.1  # Time step
    speed = 2.0  # Constant speed
    
    # Simple PID controller parameters
    k_p = 0.5  # Proportional gain
    
    # Target index
    target_idx = 0
    max_iterations = 3000
    
    # Calculate distance to end point
    end_point = np.array(control_points[-1])
    
    for i in range(max_iterations):
        # Stop if close enough to end point
        current_pos = np.array([bicycle.x, bicycle.y])
        dist_to_end = np.linalg.norm(current_pos - end_point)
        
        if dist_to_end < 2.0 and i > 100:
            break
        
        # Find closest point on path as target point
        distances = []
        search_window = 200  # Search window
        search_start = max(0, target_idx - 50)
        search_end = min(len(x_smooth), search_start + search_window)
        
        for j in range(search_start, search_end):
            point = np.array([x_smooth[j], y_smooth[j]])
            dist = np.linalg.norm(point - current_pos)
            distances.append((j, dist))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        closest_idx = distances[0][0]
        
        # Look ahead a certain distance
        look_ahead = 30
        target_idx = min(closest_idx + look_ahead, len(x_smooth) - 1)
        target_point = np.array([x_smooth[target_idx], y_smooth[target_idx]])
        
        # Calculate angle to target
        dx = target_point[0] - bicycle.x
        dy = target_point[1] - bicycle.y
        desired_theta = np.arctan2(dy, dx)
        
        # Calculate angle difference, keeping it between -pi and pi
        theta_error = (desired_theta - bicycle.theta + np.pi) % (2 * np.pi) - np.pi
        
        # Calculate steering angle
        steering_angle = np.clip(k_p * theta_error, -max_steering_angle, max_steering_angle)
        
        # Update bicycle position
        x, y, theta = bicycle.update(speed, steering_angle, dt)
        trajectory.append((x, y, theta))
    
    # Convert trajectory to array for plotting (only X and Y)
    traj_array = np.array([(t[0], t[1]) for t in trajectory])
    
    # Draw vehicle orientation arrows at intervals
    arrow_interval = max(1, len(trajectory) // 20)  # Show at most 20 arrows
    for i in range(0, len(trajectory), arrow_interval):
        x, y, theta = trajectory[i]
        dx = np.cos(theta) * 2
        dy = np.sin(theta) * 2
        ax.arrow(x, y, dx, dy, head_width=0.8, head_length=1.0, fc='r', ec='r', alpha=0.7)
    
    # Plot trajectory that satisfies bicycle constraints
    ax.plot(traj_array[:, 0], traj_array[:, 1], 'b-', linewidth=2.5, label='Bicycle-Constrained Trajectory')
    
    # Save trajectory to CSV
    save_trajectory(trajectory, "bicycle_constrained_trajectory.csv")

def save_trajectory(trajectory, filename="trajectory.csv"):
    """Save trajectory to CSV file including position and heading angle"""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y", "Theta"])
        writer.writerows(trajectory)
    print(f"Trajectory saved to {filename}")

if __name__ == "__main__":
    draw_trajectory()