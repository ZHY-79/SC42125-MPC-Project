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
from settings import *
def load_trajectory_from_csv(csv_file):  # 加载轨迹点，带有航向角
    df = pd.read_csv(csv_file)
    print(df)
    x = df['X'].values
    y = df['Y'].values
    theta = df['Theta'].values
    print(f"Loaded {len(x)} trajectory points")
    
    # Calculate theta from consecutive points
    # theta = np.zeros_like(x)
    # for i in range(len(x)-1):
    #     dx = x[i+1] - x[i]
    #     dy = y[i+1] - y[i]
    #     theta[i] = np.arctan2(dy, dx)
    # theta[-1] = theta[-2]
    
    # Create trajectory
    trajectory = []
    for i in range(len(x)):
        trajectory.append([x[i], y[i], theta[i]])
    
    return trajectory, x, y, theta

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


def CalculateAMatrix(dt, v_ref, theta_ref):
    A_d = np.eye(3)
    A_d[0, 2] = -dt * v_ref * np.sin(theta_ref)
    A_d[1, 2] = dt * v_ref * np.cos(theta_ref)
    
    return A_d

def CalculateBMatrix(dt, v_ref, theta_ref, sigma_ref, L):
    B_d = np.zeros((3, 2))
    
    B_d[0, 0] = dt * np.cos(theta_ref)
    B_d[1, 0] = dt * np.sin(theta_ref)
    B_d[2, 0] = dt * np.tan(sigma_ref) / L
    B_d[2, 1] = dt * v_ref / (L * np.cos(sigma_ref)**2)
    
    return B_d
    
def calculate_vref_and_sigma(x_ref, y_ref, theta_ref, x_next, y_next, theta_next, dt, L):

    dist = np.sqrt((x_next - x_ref)**2 + (y_next - y_ref)**2)
    v_ref = dist / dt
    d_theta = theta_next - theta_ref
    # 归一化到[-pi, pi]
    while d_theta > np.pi:
        d_theta -= 2 * np.pi
    while d_theta < -np.pi:
        d_theta += 2 * np.pi
    
    if v_ref > 1e-6:
        sigma_ref = np.arctan(L * d_theta / (v_ref * dt))
    else:
        sigma_ref = 0.0
    return v_ref, sigma_ref

def dynamicUpdate(MatrixA, MatrixB, error_k, inputerror_k):
    """
    更新离散时间动态系统的状态误差
    
    参数:
    MatrixA: 离散状态转移矩阵 A_d
    MatrixB: 离散输入矩阵 B_d
    error_k: 当前时刻的状态误差 [e_x, e_y, e_theta]
    inputerror_k: 当前时刻的输入误差 [delta_v, delta_sigma]
    
    返回:
    error_k1: 下一时刻的状态误差 [e_x, e_y, e_theta]
    """
    # 确保输入是numpy数组
    error_k = np.array(error_k).reshape(-1, 1)       # 转为列向量
    inputerror_k = np.array(inputerror_k).reshape(-1, 1)  # 转为列向量
    
    # 根据离散动态方程计算下一时刻的状态误差
    # e(k+1) = A_d * e(k) + B_d * δu(k)
    update_error = np.dot(MatrixA, error_k) + np.dot(MatrixB, inputerror_k)
    return update_error
    
def find_closest_reference_point(current_position, reference_trajectory):
    """
    找到参考轨迹上离当前位置最近的点
    
    参数:
    current_position: 当前位置 [x, y]
    reference_trajectory: 参考轨迹点列表，每个点包含 [x, y, theta]
    
    返回:
    closest_point: 最近点的索引
    """
    min_dist = float('inf')
    closest_idx = 0
    
    for i, point in enumerate(reference_trajectory):
        dist = np.sqrt((point[0] - current_position[0])**2 + 
                       (point[1] - current_position[1])**2)
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    return closest_idx



def mpc_controller(pos, v, sigma, reference_traj, N, dt, L, speed_factor=1.0):
    """
    使用CasADi实现MPC控制器 - 基于参考轨迹的速度，使用矩阵形式计算，添加终端代价
    
    参数:
    pos: 当前位置状态 [x, y, theta]
    v: 当前速度
    sigma: 当前转向角
    reference_traj: 参考轨迹，形状为(M, 3)的数组，每行包含[x, y, theta]
    N: 预测时域长度
    dt: 时间步长
    L: 车轴距离
    speed_factor: 速度因子，可以用来调整参考速度，默认1.0
    
    返回:
    v_optimal: 优化后的速度
    sigma_optimal: 优化后的转向角
    """
    import casadi as ca
    import numpy as np
    
    ref_idx = find_closest_reference_point(pos, reference_traj)                # 找到最近参考点的索引
    
    delta_v = ca.SX.sym('delta_v', N)                                          # 速度偏差符号变量
    delta_sigma = ca.SX.sym('delta_sigma', N)                                  # 转向角偏差符号变量
    opt_vars = ca.vertcat(ca.reshape(delta_v, -1, 1), ca.reshape(delta_sigma, -1, 1))  # 构建优化向量
    
    # 定义状态和控制权重
    Q = np.diag([1.0, 1.0, 0.5])                                               # 状态误差权重矩阵 [x, y, theta]
    R = np.diag([0.1, 0.05])                                                   # 控制偏差权重矩阵 [v, sigma]
    
    # 定义终端代价权重矩阵（增大状态误差权重）
    P = np.diag([5.0, 5.0, 2.5])                                               # 终端状态误差权重矩阵（比Q大）
    
    x_ref, y_ref, theta_ref = reference_traj[ref_idx]                          # 获取参考点状态
    e_x = pos[0] - x_ref                                                       # 计算x方向误差
    e_y = pos[1] - y_ref                                                       # 计算y方向误差
    e_theta = pos[2] - theta_ref                                               # 计算航向角误差
    
    while e_theta > np.pi: e_theta -= 2 * np.pi                                # 处理角度大于π的情况
    while e_theta < -np.pi: e_theta += 2 * np.pi                               # 处理角度小于-π的情况
    
    cost = 0                                                                   # 初始化总成本
    e_k = np.array([e_x, e_y, e_theta])                                        # 初始状态误差
    current_ref_idx = ref_idx                                                  # 当前参考点索引
    
    # 保存最终预测状态误差，用于后续添加终端代价
    final_state_error = None
    
    for i in range(N):
        if current_ref_idx < len(reference_traj) - 1:
            x_ref, y_ref, theta_ref = reference_traj[current_ref_idx]          # 获取当前参考状态
            
            next_ref_idx = min(current_ref_idx + 1, len(reference_traj) - 1)   # 确保索引不越界
            x_next, y_next, theta_next = reference_traj[next_ref_idx]          # 获取下一参考状态
            
            v_ref, sigma_ref = calculate_vref_and_sigma(
                x_ref, y_ref, theta_ref, x_next, y_next, theta_next, dt, L)    # 计算参考控制输入
            
            v_ref = v_ref * speed_factor                                       # 调整参考速度
            
            A_d = CalculateAMatrix(dt, v_ref, theta_ref)                       # 状态转移矩阵
            B_d = CalculateBMatrix(dt, v_ref, theta_ref, sigma_ref, L)         # 控制输入矩阵
            
            # 使用阶段性代价权重Q
            state_cost = e_k @ Q @ e_k                                         # 状态误差加权二次型
            
            u_k = np.array([delta_v[i], delta_sigma[i]])                       # 当前时刻控制偏差
            control_cost = u_k @ R @ u_k                                       # 控制偏差加权二次型
            
            cost = cost + state_cost + control_cost                            # 累加每个时刻的成本
            
            e_k = A_d @ e_k + B_d @ u_k                                        # 预测下一时刻状态误差
            current_ref_idx = next_ref_idx                                     # 向前移动参考点
            
            # 保存最终预测状态误差
            if i == N - 1:
                final_state_error = e_k
    
    # 添加终端代价
    if final_state_error is not None:
        terminal_cost = final_state_error @ P @ final_state_error              # 终端状态误差加权二次型
        cost = cost + terminal_cost                                            # 添加终端代价到总成本
    
    nlp = {'x': opt_vars, 'f': cost}                                           # 定义非线性规划问题
    
    opts = {
        'ipopt.print_level': 0,                                                # 抑制IPOPT输出
        'ipopt.max_iter': 30,                                                  # 最大迭代次数
        'ipopt.tol': 1e-4,                                                     # 收敛容差
        'ipopt.acceptable_tol': 1e-4,                                          # 可接受的容差
        'print_time': 0                                                        # 不打印计算时间
    }
    
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)                           # 创建IPOPT求解器实例
    
    lbx = []                                                                   # 下界列表
    ubx = []                                                                   # 上界列表
    
    for _ in range(N):
        lbx.extend([-1.0])                                                     # 最小速度偏差
        ubx.extend([1.0])                                                      # 最大速度偏差
    
    for _ in range(N):
        lbx.extend([-0.6])                                                     # 最小转向角偏差
        ubx.extend([0.6])                                                      # 最大转向角偏差
    
    try:
        sol = solver(x0=np.zeros(2*N), lbx=lbx, ubx=ubx)                       # 求解优化问题
        opt_sol = sol['x'].full().flatten()                                    # 提取优化结果
        
        optimal_delta_v = opt_sol[0]                                           # 提取速度偏差
        optimal_delta_sigma = opt_sol[N]                                       # 提取转向角偏差
        
        x_ref, y_ref, theta_ref = reference_traj[ref_idx]                      # 当前参考状态
        next_ref_idx = min(ref_idx + 1, len(reference_traj) - 1)               # 下一参考点索引
        x_next, y_next, theta_next = reference_traj[next_ref_idx]              # 下一参考状态
        
        v_ref, sigma_ref = calculate_vref_and_sigma(
            x_ref, y_ref, theta_ref, x_next, y_next, theta_next, dt, L)        # 计算参考控制
        
        v_ref = v_ref * speed_factor                                           # 调整参考速度
        
        v_optimal = v_ref + optimal_delta_v                                    # 参考速度加偏差
        sigma_optimal = sigma_ref + optimal_delta_sigma                        # 参考转向角加偏差
        
    except Exception as e:
        print(f"Optimization failed: {e}")                                     # 打印错误信息
        
        x_ref, y_ref, theta_ref = reference_traj[ref_idx]                      # 当前参考状态
        next_ref_idx = min(ref_idx + 1, len(reference_traj) - 1)               # 下一参考点索引
        x_next, y_next, theta_next = reference_traj[next_ref_idx]              # 下一参考状态
        
        v_ref, sigma_ref = calculate_vref_and_sigma(
            x_ref, y_ref, theta_ref, x_next, y_next, theta_next, dt, L)        # 计算参考控制
        
        v_ref = v_ref * speed_factor                                           # 调整参考速度
        
        v_optimal = v_ref                                                      # 使用参考速度
        sigma_optimal = sigma_ref                                              # 使用参考转向角
    
    return v_optimal, sigma_optimal                                           # 返回优化后的控制输入


# Add these functions to your existing code

def initialize_main_plot():
    """Initialize only the main trajectory plot"""
    # Main trajectory plot
    fig1, ax1 = initialize_plot()
    return fig1, ax1


def run_prius_with_controller(csv_file="bicycle_constrained_trajectory.csv", 
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
    trajectory, ref_x, ref_y, theta_ref = load_trajectory_from_csv(csv_file)
    
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
    
    # 设置可视化 - 仅初始化主轨迹图
    fig1, ax1 = initialize_main_plot()
    
    # 添加一些虚拟数据点，强制matplotlib使用完整范围
    ax1.scatter([-50, 50], [-50, 50], s=0, alpha=0)  # 透明点，用于扩展范围
    
    # 绘制参考轨迹
    ax1.plot(ref_x, ref_y, 'g--', linewidth=2, label='Reference Trajectory')
    
    # 绘制起点和终点
    ax1.plot(ref_x[0], ref_y[0], 'go', markersize=8, label='Start')
    ax1.plot(ref_x[-1], ref_y[-1], 'ro', markersize=8, label='End')
    
    # 绘制成功圆圈
    end_circle = plt.Circle((ref_x[-1], ref_y[-1]), 3.0, color='r', fill=False, linestyle='--', alpha=0.5)
    ax1.add_patch(end_circle)
    
    # 绘制范围框
    ax1.plot([-50, 50, 50, -50, -50], [-50, -50, 50, 50, -50], 'k-', alpha=0.2)
    
    ax1.legend()
    
    # 再次强制设置坐标轴范围
    force_axis_limits(ax1)
    
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
    
    # 设置扰动区间
    disturbance_start = 30   # 开始扰动的步数
    disturbance_end = 130    # 结束扰动的步数
    velocity_disturbance = 2.0  # 速度扰动值
    steering_disturbance = -0.8  # 转向扰动值
    
    print(f"Disturbance will be applied at steps {disturbance_start}-{disturbance_end}")
    
    # 用于记录数据的列表
    time_points = []
    velocity_inputs = []
    steering_inputs = []
    error_x_list = []
    error_y_list = []
    error_theta_list = []
    
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
            
            # 计算时间点
            current_time = i * dt
            time_points.append(current_time)
            
            # 使用MPC计算控制输入
            velocity, steering_angle = mpc_controller(pos, action[0], action[1], trajectory, 30, 0.05, 4.6)

            # 如果指定了基础速度，则覆盖计算的速度
            if constant_velocity:
                velocity = base_velocity
            
            # 应用扰动（仅在指定区间内）
            if disturbance_start <= i < disturbance_end:
                if i == disturbance_start:
                    print(f"Applying disturbance at step {i}")
                
                # 修改控制输入
                velocity += velocity_disturbance
                steering_angle += steering_disturbance
                
                # 可选：在扰动期间使用红色标记轨迹
                if i % update_interval == 0:
                    ax1.plot(actual_x[-1], actual_y[-1], 'ro', markersize=4)
            
            # 记录控制输入
            velocity_inputs.append(velocity)
            steering_inputs.append(steering_angle)
            
            # 计算误差 - 找到最近的参考点
            ref_idx = find_closest_reference_point([pos[0], pos[1]], trajectory)
            ref_pos = trajectory[ref_idx]
            
            # 计算位置和航向误差
            error_x = pos[0] - ref_pos[0]
            error_y = pos[1] - ref_pos[1]
            error_theta = pos[2] - ref_pos[2]
            
            # 规范化航向角误差到 [-pi, pi]
            while error_theta > np.pi:
                error_theta -= 2 * np.pi
            while error_theta < -np.pi:
                error_theta += 2 * np.pi
                
            # 记录误差
            error_x_list.append(error_x)
            error_y_list.append(error_y)
            error_theta_list.append(error_theta)
            
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
                update_vehicle(ax1, pos)
                
                # 绘制轨迹
                if len(actual_x) > update_interval:
                    ax1.plot(actual_x[-update_interval:], actual_y[-update_interval:], 'b-', linewidth=1.5)
                else:
                    ax1.plot(actual_x, actual_y, 'b-', linewidth=1.5)
                
                # 强制范围
                force_axis_limits(ax1)
                
                # 更新主图形
                fig1.canvas.draw_idle()
                plt.pause(0.001)
            
            # 更新进度条
            progress_bar.update(1)
            
            # 检查成功条件
            dist_to_end = np.sqrt((pos[0]-trajectory[-1][0])**2 + (pos[1]-trajectory[-1][1])**2)
            if dist_to_end < 3.0:
                print(f'Success! Distance to goal: {dist_to_end:.2f}m')
                ax1.plot(actual_x, actual_y, 'b-', linewidth=1.5, label='Actual Path')
                # 使用update_vehicle绘制最终位置的车辆，设置不同颜色
                update_vehicle(ax1, pos)
                ax1.legend()
                
                # 保存轨迹图
                force_axis_limits(ax1)
                fig1.savefig("mpc_trajectory_with_disturbance.png", bbox_inches='tight')
                
                # 创建并保存输入图
                plot_inputs_and_errors(time_points, velocity_inputs, steering_inputs, 
                                      error_x_list, error_y_list, error_theta_list)
                
                env.close()
                return True
            
            if terminated:
                break
    
    # 最终可视化
    ax1.plot(actual_x, actual_y, 'b-', linewidth=1.5, label='Actual Path')
    ax1.legend()
    
    # 保存轨迹图
    force_axis_limits(ax1)
    fig1.savefig("mpc_trajectory_with_disturbance.png", bbox_inches='tight')
    
    # 创建并保存输入图
    plot_inputs_and_errors(time_points, velocity_inputs, steering_inputs, 
                          error_x_list, error_y_list, error_theta_list)
    
    env.close()
    return False

def plot_inputs_and_errors(time_points, velocity_inputs, steering_inputs, 
                          error_x_list, error_y_list, error_theta_list):
    """
    Create and save plots for control inputs and errors at the end of simulation
    """
    # Create figure for inputs
    fig2, (ax_vel, ax_steer) = plt.subplots(2, 1, figsize=(10, 8))
    ax_vel.set_title('Velocity Input')
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('Velocity (m/s)')
    ax_vel.plot(time_points, velocity_inputs, 'b-')
    ax_vel.grid(True)
    
    ax_steer.set_title('Steering Angle Input')
    ax_steer.set_xlabel('Time (s)')
    ax_steer.set_ylabel('Steering Angle (rad)')
    ax_steer.plot(time_points, steering_inputs, 'r-')
    ax_steer.grid(True)
    
    plt.tight_layout()
    fig2.savefig("mpc_control_inputs.png", bbox_inches='tight')
    
    # Create figure for errors
    fig3, (ax_err_x, ax_err_y, ax_err_theta) = plt.subplots(3, 1, figsize=(10, 10))
    ax_err_x.set_title('Position Error X')
    ax_err_x.set_xlabel('Time (s)')
    ax_err_x.set_ylabel('Error (m)')
    ax_err_x.plot(time_points, error_x_list, 'g-')
    ax_err_x.grid(True)
    
    ax_err_y.set_title('Position Error Y')
    ax_err_y.set_xlabel('Time (s)')
    ax_err_y.set_ylabel('Error (m)')
    ax_err_y.plot(time_points, error_y_list, 'm-')
    ax_err_y.grid(True)
    
    ax_err_theta.set_title('Heading Error')
    ax_err_theta.set_xlabel('Time (s)')
    ax_err_theta.set_ylabel('Error (rad)')
    ax_err_theta.plot(time_points, error_theta_list, 'c-')
    ax_err_theta.grid(True)
    
    plt.tight_layout()
    fig3.savefig("mpc_tracking_errors.png", bbox_inches='tight')

# Make sure the initialize_plot and CoG_to_RearAxle functions are defined correctly in your code
if __name__ == "__main__":
    csv_file = "bicycle_constrained_trajectory.csv"
    
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