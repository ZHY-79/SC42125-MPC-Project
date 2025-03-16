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
import casadi as ca  # 非线性数值优化求解器
import pandas as pd
from tqdm import tqdm
import math
import time

def load_trajectory_from_csv(csv_file):  # 加载轨迹点，计算轨迹方向（航向角）
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
        
def find_target_point(pos, trajectory, target_idx, lookahead_distance=0.0):
    """寻找目标点（控制接口）"""
    found_target = False
    
    for i in range(target_idx, len(trajectory)): #从当前目标索引开始，不从头开始
        dx = trajectory[i][0] - pos[0]
        dy = trajectory[i][1] - pos[1]
        distance = math.sqrt(dx*dx + dy*dy)  #欧几里得距离表述
        
        if distance >= lookahead_distance: #对于远离视域的点加入目标点
            target_point = trajectory[i]
            target_idx = i
            found_target = True
            break
    
    # 如果没找到合适的前视点，使用最后一个点
    if not found_target:
        target_point = trajectory[-1]
        target_idx = len(trajectory) - 1
    
    return target_point, target_idx


def model_predictive_control(pos, trajectory, target_idx, N=30, dt=0.1,
                            lookahead_distance=0.0, prev_steering=0.0,
                            constant_velocity=True, base_velocity=1.0):
    """Model Predictive Control for trajectory tracking
    
    Parameters:
    pos -- current position [x, y, theta]
    trajectory -- reference trajectory points [[x, y, theta], ...]
    target_idx -- current target point index
    N -- prediction horizon length
    dt -- time step
    lookahead_distance -- lookahead distance for reference point selection
    prev_steering -- previous steering angle (for smoothing)
    constant_velocity -- whether to use constant velocity
    base_velocity -- base velocity value
    
    Returns:
    steering_angle -- computed steering angle
    velocity -- computed velocity
    target_idx -- updated target index
    target_point -- current target point
    """
    # 1. Find reference trajectory points within the prediction horizon
    ref_points = []
    current_idx = target_idx

    for i in range(N):
        # Scale down the lookahead distance for further points in the horizon
        scaled_lookahead = lookahead_distance
        target_point, current_idx = find_target_point(pos, trajectory, current_idx, 
                                                     lookahead_distance=scaled_lookahead)
        ref_points.append(target_point)

    # 2. Define constraints
    max_steering = 0.4  # ~20 degrees
    min_steering = -max_steering
    max_steering_rate = 0.3  # maximum steering change per step

    # 3. Setup the CasADi optimization problem
    # State variables
    x = ca.SX.sym('x')
    y = ca.SX.sym('y')
    theta = ca.SX.sym('theta')
    states = ca.vertcat(x, y, theta)
    n_states = states.size1()

    # Control variables  控制变量，包括速度和航向角
    v = ca.SX.sym('v')
    delta = ca.SX.sym('delta')
    controls = ca.vertcat(v, delta)
    n_controls = controls.size1()

    # Bicycle model dynamics (simplified)
    x_next = x + v * ca.cos(theta) * dt
    y_next = y + v * ca.sin(theta) * dt
    theta_next = theta + v * ca.tan(delta) / 2.86 * dt  # 2.86m is wheelbase

    # Dynamics as a CasADi function
    f = ca.Function('f', [states, controls], [ca.vertcat(x_next, y_next, theta_next)])

    # Optimization variables  #控制输入和状态的定义
    U = ca.SX.sym('U', n_controls, N)  # N control inputs 2*N
    X = ca.SX.sym('X', n_states, N+1)  # N+1 state points

    # Parameters (current state + reference trajectory + base velocity) P = [当前状态(n_states), 参考轨迹点1(n_states), 参考轨迹点2(n_states), ..., 参考轨迹点N(n_states), 基础速度(1)]
    P = ca.SX.sym('P', n_states + n_states*N + 1)

    # Initialize objective function and constraints
    obj = 0
    g = ca.vertcat()

    # Initial state constraint must equal current constraint
    g = ca.vertcat(g, X[:, 0] - P[0:n_states])  

    # Define cost matrices
    Q = np.diag([20.0, 20.0, 1.0])  # State error cost weights  x y theta
    R = np.diag([0.5, 2.0])         # Control cost weights  (velocity cost and delta) 更倾向变速而非转弯

    # Loop through the prediction horizon
    for i in range(N):
        # Get reference state for this step
        ref_idx = n_states + i*n_states
        ref = P[ref_idx:ref_idx+n_states]  # 指向从轨迹点1开始的每三个状态
        
        # State error cost
        state_error = X[:, i] - ref
        
        # Control cost (deviation from reference velocity)
        control_error = U[:, i] - ca.vertcat(P[n_states*(N+1)], 0)  # 导入基础速度，同时希望转向角为0
        
        # Add to objective function
        obj += state_error.T @ Q @ state_error + control_error.T @ R @ control_error
        
        # Add steering rate penalty for smoothness
        if i > 0:
            steering_rate = U[1, i] - U[1, i-1]  # 前后时刻转向角差异
            obj += 0.5 * steering_rate**2  # 加上该项转向角惩罚用于平滑
        
        # System dynamics constraint
        x_next = f(X[:, i], U[:, i])
        g = ca.vertcat(g, X[:, i+1] - x_next)

    # Terminal cost (higher weight on final state)  采用高额权重迫使其满足终端集合约束
    terminal_ref = P[n_states + (N-1)*n_states:n_states + N*n_states]
    terminal_error = X[:, N] - terminal_ref
    obj += 5.0 * terminal_error.T @ Q @ terminal_error

    # Create the NLP problem
    OPT_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))
    nlp = {'f': obj, 'x': OPT_variables, 'g': g, 'p': P}

    # Solver options
    opts = {
        'ipopt': {
            'max_iter': 500,
            'print_level': 0,
            'acceptable_tol': 1e-8,
            'acceptable_obj_change_tol': 1e-6,
            'hessian_approximation': 'limited-memory',  # 使用L-BFGS逼近Hessian
            'mu_strategy': 'adaptive',
            'tol': 1e-5,
            'linear_solver': 'mumps'
        },
        'print_time': 0
    }

    # Create solver
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Variable bounds
    lbx = ca.DM.zeros(n_controls*N + n_states*(N+1), 1)
    ubx = ca.DM.zeros(n_controls*N + n_states*(N+1), 1)

    # Set control bounds
    for i in range(n_controls*N):
        if i % n_controls == 0:  # Velocity
            if constant_velocity:
                # Enforce exact velocity if constant
                lbx[i] = base_velocity
                ubx[i] = base_velocity
            else:
                # Allow velocity range if adaptive
                lbx[i] = 0.3  # Min velocity
                ubx[i] = 2.0  # Max velocity
        else:  # Steering
            lbx[i] = min_steering
            ubx[i] = max_steering

    # Set state bounds (no strict bounds, can be anywhere in state space)
    for i in range(n_controls*N, n_controls*N + n_states*(N+1)):
        lbx[i] = -ca.inf
        ubx[i] = ca.inf

    # Constraint bounds (dynamics must be satisfied exactly)
    lbg = ca.DM.zeros(n_states*(N+1), 1)
    ubg = ca.DM.zeros(n_states*(N+1), 1)

    # Prepare parameters
    parameters = ca.DM.zeros(n_states + n_states*N + 1, 1)

    # Current state
    parameters[0:n_states] = ca.DM(pos)

    # Reference trajectory
    for i in range(N):
        ref_idx = min(i, len(ref_points)-1)
        parameters[n_states + i*n_states:n_states + (i+1)*n_states] = ca.DM(ref_points[ref_idx])

    # Base velocity parameter
    parameters[n_states*(N+1)] = base_velocity if constant_velocity else compute_velocity(prev_steering, constant=False, base_velocity=base_velocity)

    # Initial guess for optimization variables
    initial_x = ca.DM.zeros(n_controls*N + n_states*(N+1), 1)

    # Solve the optimization problem
    try:
        solution = solver(x0=initial_x, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg, p=parameters)
        
        # Extract solution
        x_opt = solution['x']
        
        # Get optimal controls (first control action)
        u_opt = x_opt[0:n_controls*N].reshape((n_controls, N))
        steering_angle = float(u_opt[1, 0])
        velocity = float(u_opt[0, 0])
        
        # Limit steering rate for smoothness
        steering_angle = max(prev_steering - max_steering_rate, 
                           min(prev_steering + max_steering_rate, steering_angle))
    except:
        # Fallback if optimization fails - use a simple approach
        print("MPC optimization failed, using simple fallback")
        # Simple fallback - steer towards the closest trajectory point with a proportional controller
        target_point, target_idx = find_target_point(pos, trajectory, target_idx, lookahead_distance)
        dx = target_point[0] - pos[0]
        dy = target_point[1] - pos[1]
        angle_to_target = math.atan2(dy, dx)
        angle_diff = angle_to_target - pos[2]
        while angle_diff > math.pi: angle_diff -= 2.0 * math.pi
        while angle_diff < -math.pi: angle_diff += 2.0 * math.pi
        steering_angle = max_steering * (angle_diff / math.pi)  # Simple proportional control
        steering_angle = max(min_steering, min(max_steering, steering_angle))
        velocity = base_velocity if constant_velocity else 0.5  # Conservative velocity

    # Update target point and index for visualization
    target_point, target_idx = find_target_point(pos, trajectory, target_idx, lookahead_distance)

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

def run_prius_with_controller(csv_file="smoothed_trajectory.csv", 
                            constant_velocity=True, base_velocity=1.0, 
                            lookahead_distance=5.0, dt=0.1):
    """
    Run Prius simulation with MPC trajectory tracking
    
    Parameters:
    csv_file: Path to trajectory CSV file
    constant_velocity: Whether to use constant velocity
    base_velocity: Base velocity in m/s
    lookahead_distance: Lookahead distance for reference points
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
            urdf='prius.urdf',  # 使用简化模型
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
    print(f"Starting simulation using MPC controller with {velocity_mode} velocity...")
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
            
            # 使用MPC计算控制输入
            steering_angle, velocity, target_idx, target_point = model_predictive_control(
                pos, trajectory, target_idx, N=10, dt=dt, 
                lookahead_distance=lookahead_distance, prev_steering=prev_steering, 
                constant_velocity=constant_velocity, base_velocity=base_velocity)
            
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
                plt.savefig("mpc_trajectory_success.png", bbox_inches='tight')
                env.close()
                return True
            
            if terminated:
                break
    
    # 最终可视化
    ax.plot(actual_x, actual_y, 'b-', linewidth=1.5, label='Actual Path')
    ax.legend()
    
    # 强制范围并保存
    force_axis_limits(ax)
    plt.savefig("mpc_trajectory_final.png", bbox_inches='tight')
    
    env.close()
    return False

if __name__ == "__main__":
    csv_file = "smoothed_trajectory.csv"
    
    print(f'Running trajectory from: {csv_file}')
    
    # 直接运行模拟，使用MPC控制器和恒定速度模式
    start_time = time.time()
    success = run_prius_with_controller(
        csv_file=csv_file,
        constant_velocity=False,       # 是否使用恒定速度
        base_velocity=1.0,            # 基础速度 (m/s)
        lookahead_distance=5.0,       # 前视距离 (m)
        dt=0.02                       # 仿真时间步长 (s)
    )
    elapsed_time = time.time() - start_time
    
    print(f"Result: {'Success' if success else 'Incomplete'}")
    print(f"Time: {elapsed_time:.2f} seconds")