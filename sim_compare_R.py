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
import scipy.linalg as la
import os

# 确保compare_R目录存在
if not os.path.exists("compare_R"):
    os.makedirs("compare_R")

def is_in_terminal_set(P_matrix, state, radius):
    ref_state=[0, 0, 0]
    error_x = state[0] - ref_state[0]
    error_y = state[1] - ref_state[1]
    error_theta = state[2] - ref_state[2]
    
    while error_theta > np.pi:
        error_theta -= 2 * np.pi
    while error_theta < -np.pi:
        error_theta += 2 * np.pi
    
    error_vector = np.array([error_x, error_y, error_theta])
    quadratic_form = error_vector.T @ P_matrix @ error_vector
    
    return float(quadratic_form) <= radius

def load_trajectory_from_csv(csv_file):  # 加载轨迹点，带有航向角
    df = pd.read_csv(csv_file)
    print(df)
    x = df['X'].values
    y = df['Y'].values
    theta = df['Theta'].values
    print(f"Loaded {len(x)} trajectory points")
    trajectory = []
    for i in range(len(x)):
        trajectory.append([x[i], y[i], theta[i]])
    
    return trajectory, x, y, theta

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
    
def dynamicUpdate(MatrixA, MatrixB, error_k, inputerror_k):

    # 确保输入是numpy数组
    error_k = np.array(error_k).reshape(-1, 1)       # 转为列向量
    inputerror_k = np.array(inputerror_k).reshape(-1, 1)  # 转为列向量
    
    # 根据离散动态方程计算下一时刻的状态误差
    # e(k+1) = A_d * e(k) + B_d * δu(k)
    update_error = np.dot(MatrixA, error_k) + np.dot(MatrixB, inputerror_k)
    return update_error

def find_lookahead_point(pos, sampled_trajectory, lookahead_distance):
    current_position = pos[:2]
    current_heading = pos[-1]
    
    # 计算车辆朝向的单位向量
    heading_vector = np.array([np.cos(current_heading), np.sin(current_heading)])
    
    # 在前方的点中找最近的满足前视距离的点
    min_dist = float('inf')
    result_idx = len(sampled_trajectory) - 1  # 默认为最后一个点
    
    for i, point in enumerate(sampled_trajectory):
        # 计算当前点到参考点的向量
        dx = point[0] - current_position[0]
        dy = point[1] - current_position[1]
        
        # 计算距离
        dist = np.sqrt(dx**2 + dy**2)
        
        # 计算到参考点的单位向量
        point_vector = np.array([dx, dy])
        if np.linalg.norm(point_vector) > 0:  # 避免除以零
            point_vector = point_vector / np.linalg.norm(point_vector)
        
        # 点乘为正表示参考点在车辆前方
        is_in_front = np.dot(heading_vector, point_vector) > 0
        
        # 如果点在前方且距离大于等于前视距离，判断是否更近
        if is_in_front and dist >= lookahead_distance and dist < min_dist:
            min_dist = dist
            result_idx = i
    
    # 返回找到的前视点的x、y坐标和角度
    target_point = sampled_trajectory[result_idx]
    return target_point  # 直接返回目标点的[x, y, theta]

def sample_trajectory(original_trajectory, num_segments=100):
    traj_length = len(original_trajectory)
    segment_size = max(1, traj_length // num_segments)
    
    # 第一个采样点是从segment_size开始，不是从0开始（跳过起点）
    sampled_indices = [segment_size * i for i in range(1, num_segments + 1) if segment_size * i < traj_length]
    
    # 确保包含最后一个点
    if traj_length - 1 not in sampled_indices:
        sampled_indices.append(traj_length - 1)
    print(sampled_indices)
    # 返回实际的轨迹点而不是索引
    return [original_trajectory[i] for i in sampled_indices]

def mpc_controller(pos, reference_traj, sampled_traj, N, dt, L, lookahead_distance=2.0, initial_state=None, v_ref=2, sigma_ref=0, Q_gain=5, R_gain=3):      # 注意这里vref = 0， 因为要停下来，但是无法求解

    # 如果提供了初始状态，则使用它替代当前状态
    if initial_state is not None:
        pos = initial_state
    
    x_ref, y_ref, theta_ref = find_lookahead_point(pos, sampled_traj, lookahead_distance)
    
    # 创建优化变量
    delta_v = ca.SX.sym('delta_v', N)                                          # 速度偏差符号变量
    delta_sigma = ca.SX.sym('delta_sigma', N)                                  # 转向角偏差符号变量
    opt_vars = ca.vertcat(ca.reshape(delta_v, -1, 1), ca.reshape(delta_sigma, -1, 1))  # 构建优化向量
    
    print(f'(x, y, theta) ref: {x_ref, y_ref, theta_ref}')
    # 定义状态和控制权重
    Q = np.diag([1, 1, 1])                                               # 状态误差权重矩阵 [x, y, theta]
    R = np.diag([1, 1])                                                   # 控制偏差权重矩阵 [v, sigma]
    Q = Q * Q_gain
    R = R * R_gain
    A_d = CalculateAMatrix(dt, v_ref, theta_ref)                               # 状态转移矩阵
    B_d = CalculateBMatrix(dt, v_ref, theta_ref, sigma_ref, L)                 # 控制输入矩阵
                                                                        
    P = la.solve_discrete_are(A_d, B_d, Q, R)
    beta = 15     # beta * 终端
    # 5. 计算初始状态误差
    e_x = pos[0] - x_ref                                                       # 计算x方向误差
    e_y = pos[1] - y_ref                                                       # 计算y方向误差
    e_theta = pos[2] - theta_ref                                               # 计算航向角误差
    
    # 归一化角度误差到 [-pi, pi]
    while e_theta > np.pi: e_theta -= 2 * np.pi                                # 处理角度大于π的情况
    while e_theta < -np.pi: e_theta += 2 * np.pi                               # 处理角度小于-π的情况
    
    cost = 0                                                                   # 初始化总成本
    e_k = np.array([e_x, e_y, e_theta])                                        # 初始状态误差
    
    # 保存最终预测状态误差，用于后续添加终端代价
    final_state_error = None

    # 6. 在整个预测horizon中使用固定的参考点和参考控制
    for i in range(N):
        # 计算状态和控制代价
        state_cost = e_k @ Q @ e_k                                             # 状态误差加权二次型
        u_k = np.array([delta_v[i], delta_sigma[i]])                           # 当前时刻控制偏差
        control_cost = u_k @ R @ u_k                                           # 控制偏差加权二次型
        stage_cost = state_cost + control_cost                                 # 当前时刻阶段代价
        cost = cost + stage_cost                                               # 累加每个时刻的成本

        # 预测下一时刻状态误差
        e_k = A_d @ e_k + B_d @ u_k                                            # 预测下一时刻状态误差
        
        # 保存最终预测状态误差
        if i == N - 1:
            final_state_error = e_k
    
    # 7. 添加终端代价
    if final_state_error is not None:
        terminal_cost = beta * final_state_error @ P @ final_state_error              # 终端状态误差加权二次型
        cost = cost + terminal_cost                                            # 添加终端代价到总成本
    
    # 8. 定义和求解优化问题
    nlp = {'x': opt_vars, 'f': cost}                                           # 定义非线性规划问题
    
    opts = {
        'ipopt.print_level': 0,                                                # 抑制IPOPT输出
        'ipopt.max_iter': 30,                                                  # 最大迭代次数
        'ipopt.tol': 1e-4,                                                     # 收敛容差
        'ipopt.acceptable_tol': 1e-4,                                          # 可接受的容差
        'print_time': 0                                                        # 不打印计算时间
    }
    
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)                           # 创建IPOPT求解器实例
    
    # 9. 设置优化边界
    lbx = []                                                                   # 下界列表
    ubx = []                                                                   # 上界列表
    
    for _ in range(N):
        lbx.extend([-1.0])                                                     # 最小速度偏差
        ubx.extend([1.0])                                                      # 最大速度偏差
    
    for _ in range(N):
        lbx.extend([-0.5])                                                     # 最小转向角偏差
        ubx.extend([0.5])                                                      # 最大转向角偏差
    
    # 10. 求解优化问题并提取结果
    optimal_cost = None

    sol = solver(x0=np.zeros(2*N), lbx=lbx, ubx=ubx)                       # 求解优化问题
    opt_sol = sol['x'].full().flatten()                                    # 提取优化结果
    optimal_cost = float(sol['f'])                                         # 提取最优代价值
        
    optimal_delta_v = opt_sol[0]                                           # 提取速度偏差
    optimal_delta_sigma = opt_sol[N]                                       # 提取转向角偏差
        
    # 应用优化结果计算最终控制输入
    v_optimal = v_ref + optimal_delta_v                                    # 参考速度加偏差
    sigma_optimal = sigma_ref + optimal_delta_sigma                        # 参考转向角加偏差
        
    # 计算实际使用的输入的阶段代价 (使用实际delta_v和delta_sigma)
    u_actual = np.array([optimal_delta_v, optimal_delta_sigma])            # 实际输入偏差
    actual_control_cost = u_actual @ R @ u_actual                          # 实际控制代价
    actual_stage_cost = (e_x**2 * Q[0,0] + e_y**2 * Q[1,1] + e_theta**2 * Q[2,2]) + actual_control_cost  # 实际阶段代价
    return v_optimal, sigma_optimal, optimal_cost, actual_stage_cost, optimal_delta_v, optimal_delta_sigma, P           # 返回优化后的控制输入、最优代价和当前阶段代价

# Add these functions to your existing code

def initialize_main_plot():
    """Initialize only the main trajectory plot"""
    # Main trajectory plot
    fig1, ax1 = initialize_plot()
    return fig1, ax1


def run_prius_with_controller(csv_file="bicycle_constrained_trajectory.csv", 
                            v_ref = 4,
                            lookahead_distance=5.0, dt=0.1,
                            N=30,
                            custom_start_pos=None,
                            Q_gain=5,
                            R_gain=3):

    # 加载轨迹
    trajectory, ref_x, ref_y, theta_ref = load_trajectory_from_csv(csv_file)
    # 设置最大步数
    n_steps = min(100000, len(trajectory) * 10)
    print(f"Maximum steps: {n_steps}")
    sampled_traj = sample_trajectory(trajectory)
    radius = 4
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
    
    # 设置可视化
    fig1, ax1 = initialize_main_plot()
    
    # 绘制参考轨迹
    ax1.plot(ref_x, ref_y, 'g--', linewidth=2, label='Reference Trajectory')
    
    # 绘制起点和终点
    ax1.plot(ref_x[0], ref_y[0], 'go', markersize=8, label='Trajectory Start')
    ax1.plot(ref_x[-1], ref_y[-1], 'ro', markersize=8, label='Trajectory End')
    
    ax1.legend()
    first_point = ax1.plot(sampled_traj[0][0], sampled_traj[0][1], 'x', color='blue', label='Reference Points')
    for ref in sampled_traj:
        ax1.plot(ref[0], ref[1], 'x', color='blue')
    ax1.legend()
        
    # 设置起始点
    if custom_start_pos is not None:
        if len(custom_start_pos) >= 3:
            pos0 = np.array([custom_start_pos[0], custom_start_pos[1], custom_start_pos[2], 0.0])
        else:
            print("Warning: Custom start position needs at least [x, y, theta]. Using trajectory start instead.")
            pos0 = np.array([trajectory[0][0], trajectory[0][1], trajectory[0][2], 0.0])
    else:
        pos0 = np.array([trajectory[0][0], trajectory[0][1], trajectory[0][2], 0.0])
    
    print(f"Start: X={pos0[0]:.2f}, Y={pos0[1]:.2f}, Theta={pos0[2]:.2f}")
    print(f"End: X={trajectory[-1][0]:.2f}, Y={trajectory[-1][1]:.2f}")
    
    # 在地图上标记起始点
    ax1.plot(pos0[0], pos0[1], 'mo', markersize=8, label='Vehicle Start')
    ax1.legend()
    
    # 重置仿真
    ob = env.reset(pos=pos0[0:3])
    
    # 跟踪轨迹
    actual_x = [pos0[0]]
    actual_y = [pos0[1]]
    
    action = np.array([0.0, 0.0])
    
    # 可视化更新间隔
    update_interval = max(1, len(trajectory) // 100)
    
    # 用于记录数据的列表
    time_points = []
    velocity_inputs = []
    steering_inputs = []
    error_x_list = []
    error_y_list = []
    error_theta_list = []
    
    # 记录均方根位置误差和输入误差
    rmse_list = []
    velocity_error_list = []
    steering_error_list = []
    
    # 新增: 记录最优代价和阶段代价
    optimal_cost_list = []
    stage_cost_list = []
    
    # 新增: 记录代价降低和负阶段代价
    cost_decrease_list = []
    negative_stage_cost_list = []
    
    # 上一时刻的最优代价
    prev_optimal_cost = None
    with tqdm(total=n_steps, desc="Progress") as progress_bar:
        for i in range(n_steps):
            # 步进仿真
            ob, reward, terminated, truncated, info = env.step(action)
            sigma_ref = 0
            pos = ob['robot_0']['joint_state']['position']
            pos = CoG_to_RearAxle(pos)
            
            # 跟踪实际轨迹
            actual_x.append(pos[0])
            actual_y.append(pos[1])
            
            # 计算时间点
            current_time = i * dt
            time_points.append(current_time)
            
            # 使用MPC计算控制输入以及获取最优代价和阶段代价
            velocity, steering_angle, optimal_cost, stage_cost, velocity_error, steer_error, P = mpc_controller(
                pos, trajectory, sampled_traj, N, dt, 4.6, 
                lookahead_distance=lookahead_distance, 
                v_ref=v_ref, 
                sigma_ref=0,
                Q_gain=Q_gain,
                R_gain=R_gain
            )
            
            # 记录最优代价和阶段代价
            optimal_cost_list.append(optimal_cost)
            stage_cost_list.append(stage_cost)
            
            # 计算代价降低 (当前最优代价 - 上一时刻最优代价)
            if prev_optimal_cost is not None:
                cost_decrease = optimal_cost - prev_optimal_cost
                cost_decrease_list.append(cost_decrease)
                negative_stage_cost_list.append(-stage_cost)
            
            # 更新上一时刻最优代价
            prev_optimal_cost = optimal_cost
    
            # 记录控制输入
            velocity_inputs.append(velocity)
            steering_inputs.append(steering_angle)
            
            # 计算误差 - 找到最近的参考点
            ref_pos = find_lookahead_point(pos, trajectory, lookahead_distance)
            
            # 计算位置和航向误差
            error_x = pos[0] - ref_pos[0]
            error_y = pos[1] - ref_pos[1]
            error_theta = pos[2] - ref_pos[2]
            rmse = np.sqrt(error_x**2 + error_y**2)
                
            # 计算输入误差 (实际输入与参考输入之间的差异)
            velocity_error_list.append(velocity_error)
            steering_error_list.append(steer_error)
            
            # 规范化航向角误差到 [-pi, pi]
            while error_theta > np.pi:
                error_theta -= 2 * np.pi
            while error_theta < -np.pi:
                error_theta += 2 * np.pi
                
            # 记录误差
            error_x_list.append(error_x)
            error_y_list.append(error_y)
            error_theta_list.append(error_theta)
            rmse_list.append(rmse)
            
            # 每隔一段时间显示状态
            if i % 1 == 0:
                if len(cost_decrease_list) > 0:
                    print(f"Cost Decrease: {cost_decrease_list[-1]:.4f} Stage Cost: {stage_cost:.4f}")
            
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
                
                # 更新主图形
                fig1.canvas.draw_idle()
                plt.pause(0.001)
            
            # 更新进度条
            progress_bar.update(1)
            
            # 检查成功条件
            dist_to_end = np.sqrt((pos[0]-trajectory[-1][0])**2 + (pos[1]-trajectory[-1][1])**2)
            if dist_to_end < 0.3:
                print(f'Success! Distance to goal: {dist_to_end:.2f}m')
                ax1.plot(actual_x, actual_y, 'b-', linewidth=1.5, label='Actual Path')
                update_vehicle(ax1, pos)
                ax1.legend()
                
                # 保存轨迹图
                fig1.savefig(f"compare_R/mpc_trajectory_R{R_gain}.png", bbox_inches='tight')
                
                # 创建并保存输入图
                plot_inputs_and_errors(time_points, velocity_inputs, steering_inputs, 
                                      error_x_list, error_y_list, error_theta_list, R_gain=R_gain)
                
                
                env.close()
                return True, rmse_list, error_x_list, error_y_list, error_theta_list, [velocity_error_list, steering_error_list], cost_decrease_list, negative_stage_cost_list
            
            elif pos[0] - trajectory[-1][0] >=0.5:
                env.close()
                return True, rmse_list, error_x_list, error_y_list, error_theta_list, [velocity_error_list, steering_error_list], cost_decrease_list, negative_stage_cost_list
            if terminated:
                break
    
    # 最终可视化
    ax1.plot(actual_x, actual_y, 'b-', linewidth=1.5, label='Actual Path')
    ax1.legend()
    
    # 保存轨迹图
    fig1.savefig(f"compare_R/mpc_trajectory_R{R_gain}.png", bbox_inches='tight')
    
    # 创建并保存输入图
    plot_inputs_and_errors(time_points, velocity_error_list, steering_error_list, 
                          error_x_list, error_y_list, error_theta_list, R_gain=R_gain)
    
    env.close()
    return False, rmse_list, error_x_list, error_y_list, error_theta_list, [velocity_error_list, steering_error_list], cost_decrease_list, negative_stage_cost_list

def plot_inputs_and_errors(time_points, velocity_inputs, steering_inputs, 
                          error_x_list, error_y_list, error_theta_list, Q_gain=5, R_gain=3):
    """
    Create and save plots for control inputs and errors at the end of simulation
    """
    # Create figure for inputs
    fig2, (ax_vel, ax_steer) = plt.subplots(2, 1, figsize=(10, 8))
    ax_vel.set_title(f'Velocity Error (R_gain={R_gain})')
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('Velocity (m/s)')
    ax_vel.set_ylim([-5, 5])
    ax_vel.plot(time_points, velocity_inputs, drawstyle='steps')
    ax_vel.grid(True)
    
    ax_steer.set_title(f'Steering Angle Input (R_gain={R_gain})')
    ax_steer.set_xlabel('Time (s)')
    ax_steer.set_ylabel('Steering Angle (rad)')
    ax_steer.plot(time_points, steering_inputs, drawstyle='steps')
    ax_steer.grid(True)
    
    plt.tight_layout()
    fig2.savefig(f"compare_R/mpc_control_inputs_R{R_gain}.png", bbox_inches='tight')
    
    # Create figure for errors
    fig3, (ax_err_x, ax_err_y, ax_err_theta, ax_err_dist) = plt.subplots(4, 1, figsize=(10, 10))
    ax_err_x.set_title(f'Position Error X (R_gain={R_gain})')
    ax_err_x.set_xlabel('Time (s)')
    ax_err_x.set_ylabel('Error (m)')
    ax_err_x.plot(time_points, error_x_list, drawstyle='steps')
    ax_err_x.grid(True)
    
    ax_err_y.set_title(f'Position Error Y (R_gain={R_gain})')
    ax_err_y.set_xlabel('Time (s)')
    ax_err_y.set_ylabel('Error (m)')
    ax_err_y.plot(time_points, error_y_list, drawstyle='steps')
    ax_err_y.grid(True)
    
    ax_err_theta.set_title(f'Heading Error (R_gain={R_gain})')
    ax_err_theta.set_xlabel('Time (s)')
    ax_err_theta.set_ylabel('Error (rad)')
    ax_err_theta.plot(time_points, error_theta_list, drawstyle='steps')
    ax_err_theta.grid(True)

    # 计算RMSE并显示在第四个子图中
    rmse_values = [np.sqrt(x**2 + y**2) for x, y in zip(error_x_list, error_y_list)]
    ax_err_dist.set_title(f'Position RMSE (R_gain={R_gain})')
    ax_err_dist.set_xlabel('Time (s)')
    ax_err_dist.set_ylabel('RMSE (m)')
    ax_err_dist.plot(time_points, rmse_values, drawstyle='steps')
    ax_err_dist.grid(True)

    plt.tight_layout()
    fig3.savefig(f"compare_R/mpc_tracking_errors_R{R_gain}.png", bbox_inches='tight')

# Make sure the initialize_plot and CoG_to_RearAxle functions are defined correctly in your code
if __name__ == "__main__":
    csv_file = "smooth_path_trajectory.csv"
    print(f'Running trajectory from: {csv_file}')
    
    initial_pos = [-48, -4, 0.4]  # 设计初始位置
    
    # 固定N为30，定义不同的R_gain值进行测试
    N = 30
    Q_gain = 5  # 固定Q_gain
    R_gain_values = [1, 10, 100, 1000]
    
    # 保存不同R_gain值的结果
    results = {}
    
    for R_gain in R_gain_values:
        print(f"Running simulation with R_gain = {R_gain}")
        
        success, rmse_error_list, error_x_list, error_y_list, error_theta_list, [velocity_error, sigma_error], cost_decrease_list, minus_stage_cost = run_prius_with_controller(
            csv_file=csv_file,
            v_ref=2,
            lookahead_distance=3,  # 前视距离 (m)
            dt=0.05,                 # 仿真时间步长 (s)
            N=N,
            custom_start_pos=initial_pos,
            Q_gain=Q_gain,
            R_gain=R_gain
        )
        
        # 保存该R_gain值下的结果
        results[R_gain] = {
            'rmse': rmse_error_list,
            'error_x': error_x_list,
            'error_y': error_y_list,
            'error_theta': error_theta_list,
            'velocity_error': velocity_error,
            'steering_error': sigma_error
        }
    
    # 绘制比较结果图
    plt.figure(figsize=(16, 12))
    
    # 绘制RMSE子图
    plt.subplot(4, 1, 1)
    for R_gain in R_gain_values:
        # 为每个数据系列创建对应的时间步长数组
        time_steps = np.arange(len(results[R_gain]['rmse'])) * 0.05
        plt.plot(time_steps, results[R_gain]['rmse'], label=f'R_gain = {R_gain}')
    plt.title('Position RMSE for Different R Gain Values')
    plt.xlabel('Time (s)')
    plt.ylabel('RMSE (m)')
    plt.legend()
    plt.grid(True)
    
    # 绘制X误差子图
    plt.subplot(4, 1, 2)
    for R_gain in R_gain_values:
        time_steps = np.arange(len(results[R_gain]['error_x'])) * 0.05
        plt.plot(time_steps, results[R_gain]['error_x'], label=f'R_gain = {R_gain}')
    plt.title('X Position Error for Different R Gain Values')
    plt.xlabel('Time (s)')
    plt.ylabel('X Error (m)')
    plt.legend()
    plt.grid(True)
    
    # 绘制Y误差子图
    plt.subplot(4, 1, 3)
    for R_gain in R_gain_values:
        time_steps = np.arange(len(results[R_gain]['error_y'])) * 0.05
        plt.plot(time_steps, results[R_gain]['error_y'], label=f'R_gain = {R_gain}')
    plt.title('Y Position Error for Different R Gain Values')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Error (m)')
    plt.legend()
    plt.grid(True)
    
    # 绘制theta误差子图
    plt.subplot(4, 1, 4)
    for R_gain in R_gain_values:
        time_steps = np.arange(len(results[R_gain]['error_theta'])) * 0.05
        plt.plot(time_steps, results[R_gain]['error_theta'], label=f'R_gain = {R_gain}')
    plt.title('Theta Error for Different R Gain Values')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta Error (rad)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("compare_R/mpc_comparison_different_R_gains.png", dpi=300)
    
    # 绘制控制输入对比图
    plt.figure(figsize=(16, 8))
    
    # 绘制速度输入子图
    plt.subplot(2, 1, 1)
    for R_gain in R_gain_values:
        time_steps = np.arange(len(results[R_gain]['velocity_error'])) * 0.05
        plt.plot(time_steps, results[R_gain]['velocity_error'], label=f'R_gain = {R_gain}')
    plt.title('Velocity Error for Different R Gain Values')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity Error (m/s)')
    plt.legend()
    plt.grid(True)
    
    # 绘制转向角输入子图
    plt.subplot(2, 1, 2)
    for R_gain in R_gain_values:
        time_steps = np.arange(len(results[R_gain]['steering_error'])) * 0.05
        plt.plot(time_steps, results[R_gain]['steering_error'], label=f'R_gain = {R_gain}')
    plt.title('Steering Angle Error for Different R Gain Values')
    plt.xlabel('Time (s)')
    plt.ylabel('Steering Error (rad)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("compare_R/mpc_control_inputs_comparison.png", dpi=300)