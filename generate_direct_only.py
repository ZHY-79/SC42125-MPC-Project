import matplotlib.pyplot as plt
import numpy as np
import csv

def generate_straight_line_trajectory(start_point, end_point, filename="straight_line_trajectory.csv"):
    """
    Generate a straight line trajectory from start_point to end_point
    """
    x1, y1 = start_point
    x2, y2 = end_point
    
    theta = np.arctan2(y2 - y1, x2 - x1)
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    num_points = max(10, int(distance * 5))

    trajectory = []
    for j in range(num_points + 1):
        t = j / num_points
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        trajectory.append((x, y, theta))

    save_trajectory(trajectory, filename)
    print(f"Straight line trajectory saved to {filename}")
    return trajectory

def save_trajectory(trajectory, filename="direct_traj.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y", "Theta"])
        writer.writerows(trajectory)

def draw_trajectory():
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Straight Line Trajectory: Click to Set End Point")
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.grid(True)

    # Fixed start point
    fixed_start = (-45, 0)
    ax.plot(*fixed_start, 'go', markersize=8, label='Start')

    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            end_point = (event.xdata, event.ydata)
            ax.plot(*end_point, 'ro', markersize=8, label='End')

            # Draw straight line
            ax.plot([fixed_start[0], end_point[0]], [fixed_start[1], end_point[1]], 'b-', linewidth=2, label='Trajectory')

            # Generate and save trajectory
            generate_straight_line_trajectory(fixed_start, end_point)

            ax.legend()
            fig.canvas.draw()

    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.show()

if __name__ == "__main__":
    draw_trajectory()
