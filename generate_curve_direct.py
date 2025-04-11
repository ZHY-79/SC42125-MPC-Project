def generate_straight_line_trajectory(control_points, filename="straight_line_trajectory.csv"):
    """
    Generate a straight line trajectory by sampling points along straight line segments
    connecting consecutive control points, and save to CSV file
    """
    if len(control_points) < 2:
        print("Need at least two points to generate a straight line trajectory")
        return
        
    # Store all trajectory points
    trajectory = []
    
    # Generate points along each straight line segment
    for i in range(len(control_points) - 1):
        x1, y1 = control_points[i]
        x2, y2 = control_points[i+1]
        
        # Calculate segment direction (theta)
        theta = np.arctan2(y2 - y1, x2 - x1)
        
        # Calculate distance between points
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        # Determine number of sampling points based on distance
        num_points = max(10, int(distance * 5))  # At least 10 points, or more for longer segments
        
        # Generate evenly spaced points along the segment
        for j in range(num_points + 1):
            if j == num_points and i < len(control_points) - 2:
                continue  # Skip last point except for the final segment
                
            # Linear interpolation parameter
            t = j / num_points
            
            # Calculate point coordinates
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            
            # Add point to trajectory with segment direction
            trajectory.append((x, y, theta))
    
    # Save trajectory to CSV
    save_trajectory(trajectory, filename)
    print(f"Straight line trajectory saved to {filename}")
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.interpolate import splprep, splev

def draw_trajectory():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Generate Reference Trajectory")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.grid(True)
    
    # Define start and end points
    fixed_start = (-45, 0)
    fixed_end = (5, 0)
    
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
            # Include fixed start and end points for path generation
            path_points = [fixed_start] + control_points + [fixed_end]
            
            # Draw straight lines using all points for visualization
            draw_straight_lines(ax, path_points)
            
            # Save only user-selected control points and end point (no start point)
            save_control_points(fixed_start, control_points + [fixed_end], "control_points.csv")
            
            # Generate and save straight line trajectory
            generate_straight_line_trajectory(path_points, "straight_line_trajectory.csv")
            
            # Generate smooth path trajectory using all points
            generate_smooth_path_trajectory(ax, path_points)
            
            # Add legend
            ax.legend()
            fig.canvas.draw()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()

def draw_straight_lines(ax, control_points):
    """
    Simply draw straight lines connecting the control points
    """
    # Extract x and y coordinates
    x_vals = [p[0] for p in control_points]
    y_vals = [p[1] for p in control_points]
    
    # Plot straight line segments
    ax.plot(x_vals, y_vals, 'm-', linewidth=2.5, label='Straight-Line Connection')

def save_control_points(fixed_start, control_points, filename="control_points.csv"):
    """
    Save user-selected control points with theta:
    - First point: theta from fixed start to first point
    - Other points: theta from previous point to current point
    """
    if not control_points:
        print("No control points to save.")
        return
    
    result_points = []
    
    # Process each control point
    for i in range(len(control_points)):
        x, y = control_points[i]
        
        # For first point, use direction from fixed start
        if i == 0:
            prev_x, prev_y = fixed_start
        # For all other points, use direction from previous control point
        else:
            prev_x, prev_y = control_points[i-1]
        
        # Calculate theta as direction from previous/start point to this point
        theta = np.arctan2(y - prev_y, x - prev_x)
        
        # Debug output
        if i < 2:  # Show for first two points
            point_num = i + 1
            print(f"点 {point_num}: ({x}, {y})")
            print(f"  前一点: ({prev_x}, {prev_y})")
            print(f"  theta = {theta:.4f} 弧度 ({np.degrees(theta):.2f} 度)")
        
        result_points.append((x, y, theta))
    
    # Save to CSV
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y", "Theta"])
        writer.writerows(result_points)
    print(f"Control points saved to {filename}")

def generate_smooth_path_trajectory(ax, control_points):
    """
    Generate a smooth path trajectory using spline interpolation and save it to CSV
    """
    # Use spline interpolation to generate smooth path
    cp_array = np.array(control_points)
    tck, u = splprep([cp_array[:,0], cp_array[:,1]], s=50.0, k=min(3, len(cp_array)-1))
    u_fine = np.linspace(0, 1, 1000)
    x_smooth, y_smooth = splev(u_fine, tck)
    
    # Plot smooth path
    ax.plot(x_smooth, y_smooth, 'g-', linewidth=2.5, label='Smooth Path Trajectory')
    
    # Calculate path tangent for heading angle
    dx = np.gradient(x_smooth)
    dy = np.gradient(y_smooth)
    theta = np.arctan2(dy, dx)
    
    # Create trajectory with position and heading
    smooth_trajectory = [(x_smooth[i], y_smooth[i], theta[i]) for i in range(len(x_smooth))]
    
    # Save smooth trajectory to CSV
    save_trajectory(smooth_trajectory, "smooth_multi_point_traj.csv")

def save_trajectory(trajectory, filename="direct_multi_point_traj.csv"):
    """Save trajectory to CSV file including position and heading angle"""
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y", "Theta"])
        writer.writerows(trajectory)
    print(f"Trajectory saved to {filename}")

if __name__ == "__main__":
    draw_trajectory()