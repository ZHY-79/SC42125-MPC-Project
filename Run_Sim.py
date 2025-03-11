"""
恒速版Prius轨迹跟踪模拟。
使用update_vehicle函数绘制车辆
主要特点:
1. 提供速度和角度控制接口
2. 默认使用恒定速度
3. 绘图范围固定为-50到50
"""

import numpy as np
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from bicycle_model_self import BicycleModelSelf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, Polygon
import pandas as pd
from tqdm import tqdm
import math
import time

def load_trajectory_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    
    x = df['X'].values
    y = df['Y'].values
    
    print(f"Loaded {len(x)} trajectory points")
    
    # Calculate theta from consecutive points
    theta = np.zeros_like(x)
    for i in range(len(x)-1):
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        theta[i] = np.arctan2(dy, dx)
    theta[-1] = theta[-2]
    
    # Create trajectory
    trajectory = []
    for i in range(len(x)):
        trajectory.append([x[i], y[i], theta[i]])
    
    return trajectory, x, y

def compute_steering_angle(pos, target_point, lookahead_distance=2.5, prev_steering=0.0):
    """计算转向角度（控制接口）"""
    # 计算到目标的角度
    alpha = math.atan2(target_point[1] - pos[1], target_point[0] - pos[0]) - pos[2]
    
    # 归一化角度
    while alpha > math.pi: alpha -= 2.0 * math.pi
    while alpha < -math.pi: alpha += 2.0 * math.pi
    
    # 计算转向角
    new_steering = math.atan2(2.0 * 2.86 * math.sin(alpha), lookahead_distance)
    
    # 限制转向角
    max_steering = 0.35  # 约20度
    new_steering = max(-max_steering, min(max_steering, new_steering))
    
    # 平滑转向角处理
    steering_angle = 0.6 * new_steering + 0.4 * prev_steering
    
    return steering_angle

def compute_velocity(steering_angle, constant=True, base_velocity=1.0):
    """计算速度（控制接口）"""
    if constant:
        # 恒定速度模式
        return base_velocity
    else:
        # 自适应速度模式（根据转向角调整）
        max_steering = 0.35
        turn_factor = 1.0 - 0.8 * (abs(steering_angle) / max_steering)
        velocity = base_velocity * turn_factor
        return max(0.3, velocity)  # 最小速度0.3 m/s

def find_target_point(pos, trajectory, target_idx, lookahead_distance=2.5):
    """寻找目标点（控制接口）"""
    found_target = False
    
    for i in range(target_idx, len(trajectory)):
        dx = trajectory[i][0] - pos[0]
        dy = trajectory[i][1] - pos[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance >= lookahead_distance:
            target_point = trajectory[i]
            target_idx = i
            found_target = True
            break
    
    # 如果没找到合适的前视点，使用最后一个点
    if not found_target:
        target_point = trajectory[-1]
        target_idx = len(trajectory) - 1
    
    return target_point, target_idx

def pure_pursuit_control(pos, trajectory, target_idx, lookahead_distance=2.5, prev_steering=0.0, constant_velocity=True):
    """纯跟踪控制器（使用控制接口）"""
    # 寻找目标点
    target_point, target_idx = find_target_point(pos, trajectory, target_idx, lookahead_distance)
    
    # 计算转向角
    steering_angle = compute_steering_angle(pos, target_point, lookahead_distance, prev_steering)
    
    # 计算速度
    velocity = compute_velocity(steering_angle, constant_velocity)
    
    return steering_angle, velocity, target_idx, target_point

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
    ax.set_title('Vehicle Trajectory Tracking')
    
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

def run_prius_with_controller(csv_file="smoothed_trajectory.csv", controller_type="pure_pursuit", 
                            constant_velocity=True, base_velocity=1.0, 
                            lookahead_distance=5.0, dt=0.05):
    """
    Run Prius simulation with trajectory tracking
    
    Parameters:
    csv_file: Path to trajectory CSV file
    controller_type: Type of controller ("pure_pursuit")
    constant_velocity: Whether to use constant velocity
    base_velocity: Base velocity in m/s
    lookahead_distance: Lookahead distance for pure pursuit
    dt: Simulation time step
    """
    # 加载轨迹
    trajectory, ref_x, ref_y = load_trajectory_from_csv(csv_file)
    
    # 设置最大步数
    n_steps = min(100000, len(trajectory) * 10)
    print(f"Maximum steps: {n_steps}")
    
    # 设置机器人
    robots = [
        BicycleModelSelf(
            urdf='prius.urdf',
            mode="vel",
            scaling=1,
            wheel_radius=0.31265,
            wheel_distance=2.86,
            spawn_offset=np.array([0.0, 0.0, 0.0]),
            actuated_wheels=['front_right_wheel_joint', 'front_left_wheel_joint', 'rear_right_wheel_joint', 'rear_left_wheel_joint'],
            steering_links=['front_right_steer_joint', 'front_left_steer_joint'],
            facing_direction='-x'
        )
    ]

    # 创建环境
    env = UrdfEnv(dt=dt, robots=robots, render=False)
    
    # 设置可视化 - 固定范围
    fig, ax = initialize_plot()
    
    # 添加一些虚拟数据点，强制matplotlib使用完整范围
    ax.scatter([-50, 50], [-50, 50], s=0, alpha=0)  # 透明点，用于扩展范围
    
    # 绘制参考轨迹
    ax.plot(ref_x, ref_y, 'g--', linewidth=2, label='Reference Trajectory')
    
    # 绘制起点和终点
    ax.plot(ref_x[0], ref_y[0], 'go', markersize=8, label='Start')
    ax.plot(ref_x[-1], ref_y[-1], 'ro', markersize=8, label='End')
    
    # 绘制成功圆圈
    end_circle = plt.Circle((ref_x[-1], ref_y[-1]), 3.0, color='r', fill=False, linestyle='--', alpha=0.5)
    ax.add_patch(end_circle)
    
    # 绘制范围框
    ax.plot([-50, 50, 50, -50, -50], [-50, -50, 50, 50, -50], 'k-', alpha=0.2)
    
    ax.legend()
    
    # 再次强制设置坐标轴范围
    force_axis_limits(ax)
    
    # 设置起始点
    pos0 = np.array([trajectory[0][0], trajectory[0][1], trajectory[0][2], 0.0])
    print(f"Start: X={pos0[0]:.2f}, Y={pos0[1]:.2f}")
    print(f"End: X={trajectory[-1][0]:.2f}, Y={trajectory[-1][1]:.2f}")
    
    # 重置仿真
    ob = env.reset(pos=pos0[0:3])
    
    # 跟踪轨迹
    actual_x = [pos0[0]]
    actual_y = [pos0[1]]
    
    # 初始化控制器
    target_idx = 0
    action = np.array([0.0, 0.0])
    prev_steering = 0.0
    
    # 可视化更新间隔
    update_interval = max(1, len(trajectory) // 200)
    
    # 速度模式提示
    velocity_mode = "Constant" if constant_velocity else "Adaptive"
    print(f"Starting simulation using {controller_type} controller with {velocity_mode} velocity...")
    print(f"Base velocity: {base_velocity} m/s, Lookahead distance: {lookahead_distance} m")
    
    with tqdm(total=n_steps, desc="Progress") as progress_bar:
        for i in range(n_steps):
            # 步进仿真
            ob, reward, terminated, truncated, info = env.step(action)
            
            # 获取车辆位置
            pos = ob['robot_0']['joint_state']['position']
            pos = CoG_to_RearAxle(pos)
            
            # 跟踪实际轨迹
            actual_x.append(pos[0])
            actual_y.append(pos[1])
            
            # 根据选择的控制器计算控制输入
            if controller_type == "pure_pursuit":
                steering_angle, velocity, target_idx, target_point = pure_pursuit_control(
                    pos, trajectory, target_idx, lookahead_distance, prev_steering, constant_velocity)
            # 此处可以添加其他控制器
            else:
                print(f"Unknown controller type: {controller_type}")
                env.close()
                return False
            
            # 如果指定了基础速度，则覆盖计算的速度
            if constant_velocity:
                velocity = base_velocity
            
            # 记录当前转向角
            prev_steering = steering_angle
            
            # 每10%显示进度
            progress_percent = (target_idx / (len(trajectory)-1)) * 100
            if i % (n_steps // 10) == 0:
                print(f"Progress: {progress_percent:.1f}%")
                print(f"Current position: X={pos[0]:.2f}, Y={pos[1]:.2f}")
                print(f"Control: velocity={velocity:.2f} m/s, steering={steering_angle:.4f}")
            
            # 设置下一步动作
            action = np.array([velocity, steering_angle])
            
            # 更新可视化
            if i % update_interval == 0:
                # 使用update_vehicle函数绘制车辆
                update_vehicle(ax, pos)
                
                # 绘制轨迹
                if len(actual_x) > update_interval:
                    ax.plot(actual_x[-update_interval:], actual_y[-update_interval:], 'b-', linewidth=1.5)
                else:
                    ax.plot(actual_x, actual_y, 'b-', linewidth=1.5)
                
                # 绘制目标点
                if hasattr(ax, 'target_point_marker'):
                    ax.target_point_marker.remove()
                ax.target_point_marker = ax.plot(target_point[0], target_point[1], 'rx', markersize=4)[0]
                
                # 强制范围
                force_axis_limits(ax)
                plt.pause(0.001)
            
            # 更新进度条
            progress_bar.update(1)
            
            # 检查成功条件
            dist_to_end = np.sqrt((pos[0]-trajectory[-1][0])**2 + (pos[1]-trajectory[-1][1])**2)
            if dist_to_end < 3.0:
                print(f'Success! Distance to goal: {dist_to_end:.2f}m')
                ax.plot(actual_x, actual_y, 'b-', linewidth=1.5, label='Actual Path')
                # 使用update_vehicle绘制最终位置的车辆，设置不同颜色
                update_vehicle(ax, pos)
                ax.legend()
                
                # 强制范围并保存
                force_axis_limits(ax)
                plt.savefig("trajectory_success.png", bbox_inches='tight')
                env.close()
                return True
            
            if terminated:
                break
    
    # 最终可视化
    ax.plot(actual_x, actual_y, 'b-', linewidth=1.5, label='Actual Path')
    ax.legend()
    
    # 强制范围并保存
    force_axis_limits(ax)
    plt.savefig("trajectory_final.png", bbox_inches='tight')
    
    env.close()
    return False

if __name__ == "__main__":
    csv_file = "smoothed_trajectory.csv"
    
    print(f'Running trajectory from: {csv_file}')
    
    # 直接运行模拟，使用纯跟踪控制器和恒定速度模式
    start_time = time.time()
    success = run_prius_with_controller(
        csv_file=csv_file,
        controller_type="pure_pursuit",  # 控制器类型
        constant_velocity=True,          # 是否使用恒定速度
        base_velocity=1.0,               # 基础速度 (m/s)
        lookahead_distance=5.0,          # 前视距离 (m)
        dt=0.02                          # 仿真时间步长 (s)
    )
    elapsed_time = time.time() - start_time
    
    print(f"Result: {'Success' if success else 'Incomplete'}")
    print(f"Time: {elapsed_time:.2f} seconds")