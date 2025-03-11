import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.interpolate import splprep, splev

def draw_trajectory():
    fig, ax = plt.subplots()
    ax.set_title("Draw your trajectory")
    ax.set_xlim(-50, 50)  # 调整X轴范围
    ax.set_ylim(-50, 50)  # 调整Y轴范围
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    
    trajectory = []
    fixed_start = (-46, 46)  # 设定固定起点
    fixed_end = (46, -46)  # 设定固定终点
    
    ax.plot(*fixed_start, 'go', markersize=8, label='Start Point')
    ax.plot(*fixed_end, 'ro', markersize=8, label='End Point')
    ax.legend()
    
    def on_click(event):
        if event.xdata is not None and event.ydata is not None:
            trajectory.append((event.xdata, event.ydata))
            ax.plot(event.xdata, event.ydata, 'ro')
            fig.canvas.draw()
    
    def on_key(event):
        if event.key == 'enter' and len(trajectory) > 2:
            trajectory.insert(0, fixed_start)  # 确保起点固定
            trajectory.append(fixed_end)  # 确保终点固定
            save_trajectory(trajectory)
            plot_smooth_trajectory(ax, trajectory)
            fig.canvas.draw()
    
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    plt.show()

def plot_smooth_trajectory(ax, trajectory):
    trajectory = np.array(trajectory)
    if len(trajectory) > 3:
        tck, u = splprep([trajectory[:,0], trajectory[:,1]], s=2.0, per=False)
        u_fine = np.linspace(0, 1, 5000)  # 增加采样点数
        smooth_x, smooth_y = splev(u_fine, tck)
        ax.plot(smooth_x, smooth_y, 'b-', linewidth=2, label='Smoothed Trajectory')
        save_trajectory(list(zip(smooth_x, smooth_y)), filename="smoothed_trajectory.csv")
    else:
        ax.plot(trajectory[:,0], trajectory[:,1], 'b-', linewidth=2, label='Raw Trajectory')
    ax.legend()

def save_trajectory(trajectory, filename="trajectory.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["X", "Y"])
        writer.writerows(trajectory)
    print(f"Trajectory saved to {filename}")

if __name__ == "__main__":
    draw_trajectory()
