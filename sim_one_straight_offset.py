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
from matplotlib.patches import Rectangle, FancyArrow, Polygon, Ellipse
import casadi as ca  # 非线性数值优化求解器
import pandas as pd
from tqdm import tqdm
from settings import *
import scipy.linalg as la

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

def create_single_point_trajectory():
    """创建单点轨迹 [0,0,0]"""
    trajectory = [[10, 10, 0.4]]  # 单点轨迹 [x, y, theta]
    x = trajectory[0][0]
    y = trajectory[0][1]
    theta = trajectory[0][2]
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

def mpc_controller(pos, reference_traj, N, dt, L, initial_state=None, v_ref=4, sigma_ref=0, Q=np.diag([1, 1, 1]), R=np.diag([1, 1])):
    # 如果提供了初始状态，则使用它替代当前状态
    if initial_state is not None:
        pos = initial_state
    
    # 使用单点目标轨迹
    x_ref, y_ref, theta_ref = reference_traj[0]
    print(f"ref {x_ref, y_ref, theta_ref}")
    # 创建优化变量
    delta_v = ca.SX.sym('delta_v', N)                                          # 速度偏差符号变量
    delta_sigma = ca.SX.sym('delta_sigma', N)                                  # 转向角偏差符号变量
    opt_vars = ca.vertcat(ca.reshape(delta_v, -1, 1), ca.reshape(delta_sigma, -1, 1))  # 构建优化向量
    
    # 定义状态和控制权重
    # Q = np.diag([1, 1, 1])                                               # 状态误差权重矩阵 [x, y, theta]
    # R = np.diag([1, 1])                                                   # 控制偏差权重矩阵 [v, sigma]
    A_d = CalculateAMatrix(dt, v_ref, theta_ref)                               # 状态转移矩阵
    B_d = CalculateBMatrix(dt, v_ref, theta_ref, sigma_ref, L)                 # 控制输入矩阵
                                                                             
    P = la.solve_discrete_are(A_d, B_d, Q, R)
    beta = 15         # beta * 终端
    
    # 计算初始状态误差
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

    # 在整个预测horizon中使用固定的参考点和参考控制
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
    
    # 添加终端代价
    terminal_cost = 0
    if final_state_error is not None:
        terminal_cost = beta * final_state_error @ P @ final_state_error       # 终端状态误差加权二次型
        cost = cost + terminal_cost                                            # 添加终端代价到总成本
    
    # 定义和求解优化问题
    nlp = {'x': opt_vars, 'f': cost}                                           # 定义非线性规划问题
    
    opts = {
        'ipopt.print_level': 0,                                                # 抑制IPOPT输出
        'ipopt.max_iter': 30,                                                  # 最大迭代次数
        'ipopt.tol': 1e-4,                                                     # 收敛容差
        'ipopt.acceptable_tol': 1e-4,                                          # 可接受的容差
        'print_time': 0                                                        # 不打印计算时间
    }
    
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)                           # 创建IPOPT求解器实例
    
    # 设置优化边界
    lbx = []                                                                   # 下界列表
    ubx = []                                                                   # 上界列表
    
    for _ in range(N):
        lbx.extend([-1.0])                                                     # 最小速度偏差
        ubx.extend([1.0])                                                      # 最大速度偏差
    
    for _ in range(N):
        lbx.extend([-0.5])                                                     # 最小转向角偏差
        ubx.extend([0.5])                                                      # 最大转向角偏差
    
    # 求解优化问题并提取结果
    optimal_cost = None

    sol = solver(x0=np.zeros(2*N), lbx=lbx, ubx=ubx)                           # 求解优化问题
    opt_sol = sol['x'].full().flatten()                                        # 提取优化结果
    optimal_cost = float(sol['f'])                                             # 提取最优代价值
        
    optimal_delta_v = opt_sol[0]                                               # 提取速度偏差
    optimal_delta_sigma = opt_sol[N]                                           # 提取转向角偏差
        
    # 应用优化结果计算最终控制输入
    v_optimal = v_ref + optimal_delta_v                                        # 参考速度加偏差
    sigma_optimal = sigma_ref + optimal_delta_sigma                            # 参考转向角加偏差
        
    # 计算实际使用的输入的阶段代价 (使用实际delta_v和delta_sigma)
    u_actual = np.array([optimal_delta_v, optimal_delta_sigma])                # 实际输入偏差
    actual_control_cost = u_actual @ R @ u_actual                              # 实际控制代价
    actual_stage_cost = (e_x**2 * Q[0,0] + e_y**2 * Q[1,1] + e_theta**2 * Q[2,2]) + actual_control_cost  # 实际阶段代价
    
    # 计算当前状态的终端代价 V_f(x)
    current_error = np.array([e_x, e_y, e_theta])
    current_terminal_cost = beta * current_error @ P @ current_error
    
    # 计算下一状态的终端代价 V_f(f(x,u))
    next_error = A_d @ current_error + B_d @ u_actual
    next_terminal_cost = beta * next_error @ P @ next_error

    return v_optimal, sigma_optimal, optimal_cost, actual_stage_cost, current_terminal_cost, next_terminal_cost, P

def initialize_main_plot():
    """Initialize only the main trajectory plot"""
    # Main trajectory plot
    fig1, ax1 = initialize_plot()
    return fig1, ax1

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
    fig2.savefig("origin_imgs/mpc_control_inputs_single_point.png", bbox_inches='tight')
    
    # Create figure for errors
    fig3, (ax_err_x, ax_err_y, ax_err_theta) = plt.subplots(3, 1, figsize=(10, 10))
    ax_err_x.set_title('Position Error X')
    ax_err_x.set_xlabel('Time (s)')
    ax_err_x.set_ylabel('Error (m)')
    ax_err_x.plot(time_points, error_x_list, drawstyle='steps')
    ax_err_x.grid(True)
    
    ax_err_y.set_title('Position Error Y')
    ax_err_y.set_xlabel('Time (s)')
    ax_err_y.set_ylabel('Error (m)')
    ax_err_y.plot(time_points, error_y_list, drawstyle='steps')
    ax_err_y.grid(True)
    
    ax_err_theta.set_title('Heading Error')
    ax_err_theta.set_xlabel('Time (s)')
    ax_err_theta.set_ylabel('Error (rad)')
    ax_err_theta.plot(time_points, error_theta_list, drawstyle='steps')
    ax_err_theta.grid(True)
    
    plt.tight_layout()
    fig3.savefig("offset_imgs/mpc_tracking_errors_single_point.png", bbox_inches='tight')

def run_prius_with_controller(v_ref=4, dt=0.1, N=30, custom_start_pos=None, Q=np.diag([1, 1, 1]), R=np.diag([1, 1])):
    # 使用单点轨迹
    trajectory, ref_x, ref_y, theta_ref = create_single_point_trajectory()
    
    # 设置最大步数
    n_steps = 1000  # 由于只有一个目标点，限制最大步数
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
    
    # 设置可视化
    fig1, ax1 = initialize_main_plot()
    
    # 绘制参考轨迹点
    x_points = [-12,10]  # x坐标列表
    y_points = [0, 10]  # y坐标列表

    # 画出连接这两个点的线
    
    ax1.plot(ref_x, ref_y, 'go', markersize=8, label='Target Point')
    ax1.legend()
    ax1.plot(x_points, y_points, label="Reference Trajectory")
    ax1.set_ylim([-10, 10])
    # 设置起始点
    if custom_start_pos is not None:
        if len(custom_start_pos) >= 3:
            pos0 = np.array([custom_start_pos[0], custom_start_pos[1], custom_start_pos[2], 0.0])
        else:
            print("Warning: Custom start position needs at least [x, y, theta]. Using default start position.")
            pos0 = np.array([-45, 0, 0, 0.0])
    else:
        pos0 = np.array([-45, 0, 0, 0.0])
    
    print(f"Start: X={pos0[0]:.2f}, Y={pos0[1]:.2f}, Theta={pos0[2]:.2f}")
    print(f"Target: X={trajectory[0][0]:.2f}, Y={trajectory[0][1]:.2f}")
    
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
    update_interval = 5
    
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
    
    # 记录终端代价差和负阶段代价，用于验证 V_f(f(x,u))-V_f(x) <= -stage_cost
    terminal_cost_decrease_list = []  # V_f(f(x,u))-V_f(x)
    negative_stage_cost_list = []     # -stage_cost
    
    # 终端集合参数
    terminal_P = None  # 将从MPC控制器获取
    terminal_radius = 5
    
    # 记录P的特征值
    min_eigen_P = np.inf
    max_eigen_P = 0

    with tqdm(total=n_steps, desc="Progress") as progress_bar:
        for i in range(n_steps):
            # 步进仿真
            ob, _, terminated, _, _ = env.step(action)
            sigma_ref = 0
            pos = ob['robot_0']['joint_state']['position']
            pos = CoG_to_RearAxle(pos)
            
            # 跟踪实际轨迹
            actual_x.append(pos[0])
            actual_y.append(pos[1])
            
            # 计算时间点
            current_time = i * dt
            time_points.append(current_time)
            
            # 使用修改后的MPC控制器，现在返回当前终端代价和下一步终端代价
            velocity, steering_angle, optimal_cost, stage_cost, current_terminal_cost, next_terminal_cost, P_matrix = mpc_controller(
                pos, trajectory, N, dt, 4.6, v_ref=v_ref, sigma_ref=0, Q=Q, R=R
            )
            
            # 第一次迭代时保存P矩阵
            if i == 0 and terminal_P is None:
                terminal_P = P_matrix
            
            

            eigen_v = np.linalg.eigvals(terminal_P)
            min_eigen_P = min(min_eigen_P, np.min(eigen_v))
            max_eigen_P = max(max_eigen_P, np.max(eigen_v))
            # 计算终端代价差值 V_f(f(x,u))-V_f(x)
            terminal_cost_decrease = next_terminal_cost - current_terminal_cost
            
            print(is_in_terminal_set(terminal_P, pos, terminal_radius))
            # 计算负阶段代价
            negative_stage_cost = -stage_cost
            
            
            # 检测是否在终端集合内
            if terminal_P is not None and is_in_terminal_set(terminal_P, pos, radius=terminal_radius):
                print(f"V_f(f(x,u))-V_f(x) = {terminal_cost_decrease:.6f}, -stage_cost = {negative_stage_cost:.6f}")
                print(f"Condition satisfied: {terminal_cost_decrease <= negative_stage_cost}")
                terminal_cost_decrease_list.append(terminal_cost_decrease)
                negative_stage_cost_list.append(negative_stage_cost)
            
            # 记录控制输入
            velocity_inputs.append(velocity)
            steering_inputs.append(steering_angle)
            
            # 计算误差 - 固定目标为原点 (0,0,0)
            error_x = pos[0] - trajectory[0][0]
            error_y = pos[1] - trajectory[0][1]
            error_theta = pos[2] - trajectory[0][2]
            
            # 计算均方根位置误差
            rmse = np.sqrt(error_x**2 + error_y**2)
            rmse_list.append(rmse)
            
            # 计算输入误差
            velocity_error = velocity - v_ref
            steering_error = steering_angle - sigma_ref
            velocity_error_list.append(velocity_error)
            steering_error_list.append(steering_error)
            
            # 规范化航向角误差到 [-pi, pi]
            while error_theta > np.pi:
                error_theta -= 2 * np.pi
            while error_theta < -np.pi:
                error_theta += 2 * np.pi
                
            # 记录误差
            error_x_list.append(error_x)
            error_y_list.append(error_y)
            error_theta_list.append(error_theta)
            
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
            dist_to_end = np.sqrt((pos[0]-trajectory[0][0])**2 + (pos[1]-trajectory[0][1])**2)
            print(dist_to_end)
            if dist_to_end < 0.01 or velocity<=0.001: #  and abs(pos[2]-trajectory[0][2]) < 0.1:
                print(f'Success! Distance to goal: {dist_to_end:.2f}m, Angle error: {abs(pos[2]-trajectory[0][2]):.2f}rad')
                ax1.plot(actual_x, actual_y, 'b-', linewidth=1.5, label='Actual Path')
                update_vehicle(ax1, pos)
                ax1.legend()
                
                # 保存轨迹图
                fig1.savefig("offset_imgs/mpc_trajectory_single_point.png", bbox_inches='tight')
                
                # 创建并保存输入图
                plot_inputs_and_errors(time_points, velocity_inputs, steering_inputs, 
                                      error_x_list, error_y_list, error_theta_list)
                print(f'The max eigen value of P is {max_eigen_P}, the min eigen value of P is {min_eigen_P}')
                env.close()
                return True, rmse_list, error_x_list, error_y_list, error_theta_list, [velocity_error_list, steering_error_list], terminal_cost_decrease_list, negative_stage_cost_list
            
            if terminated:
                break
    
    # 最终可视化
    ax1.plot(actual_x, actual_y, 'b-', linewidth=1.5, label='Actual Path')
    ax1.legend()
    
    # 保存轨迹图
    fig1.savefig("offset_imgs/mpc_trajectory_single_point.png", bbox_inches='tight')
    
    
    # 创建并保存输入图
    plot_inputs_and_errors(time_points, velocity_inputs, steering_inputs, 
                          error_x_list, error_y_list, error_theta_list)
    
    env.close()
    return False, rmse_list, error_x_list, error_y_list, error_theta_list, [velocity_error_list, steering_error_list], terminal_cost_decrease_list, negative_stage_cost_list


if __name__ == "__main__":
    print('Running simulation to target point [0,0,0] with different prediction horizons N')

    initial_pos = [-10, -3, 0.5]        # 设定初始位置
    N_values = [30]       # 测试不同的N值
    gain_Q = 1
    gain_P = 1
    Q = gain_Q * np.diag([5, 5, 5])   # 状态误差权重矩阵 [x, y, theta]
    R = gain_P * np.diag([5, 5])      # 控制输入权重矩阵
    
    # 存储不同N值的结果
    results = {}
    
    # 对每个N值运行仿真
    for N in N_values:
        print(f"Running simulation with N = {N}")
        success, state_error, error_x_list, error_y_list, error_theta_list, [velocity_error, sigma_error], cost_decrease_list, minus_stage_cost = run_prius_with_controller(
            v_ref=0.01,
            dt=0.05,                     # 仿真时间步长 (s)
            N=N,
            custom_start_pos=initial_pos,
            Q=Q,
            R=R
        )
        
        # 存储结果
        results[N] = {
            'success': success,
            'error_x': error_x_list,
            'error_y': error_y_list,
            'error_theta': error_theta_list,
            'velocity_error': velocity_error,
            'sigma_error': sigma_error
        }
        
        print(f"Simulation with N = {N}: {'Success' if success else 'Incomplete'}")
        print(f"Data length for N = {N}: {len(error_x_list)} steps")
    
    # 创建三个独立的图表，每个图表绘制一个状态变量的误差
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    
    # 设置颜色和线型，使图形更容易区分
    colors = ['b', 'r', 'g', 'purple']
    line_styles = ['-', '--', ':', '-.']
    
    # 绘制x误差子图
    for i, N in enumerate(N_values):
        time_steps_N = range(len(results[N]['error_x']))
        axs[0].plot(time_steps_N, results[N]['error_x'], 
                   color=colors[i % len(colors)], 
                   linestyle=line_styles[i % len(line_styles)],
                   label=f'N = {N}')
    axs[0].set_title('X Position Error')
    axs[0].set_ylabel('Error (m)')
    axs[0].grid(True)
    axs[0].legend()
    
    # 绘制y误差子图
    for i, N in enumerate(N_values):
        time_steps_N = range(len(results[N]['error_y']))
        axs[1].plot(time_steps_N, results[N]['error_y'], 
                   color=colors[i % len(colors)], 
                   linestyle=line_styles[i % len(line_styles)],
                   label=f'N = {N}')
    axs[1].set_title('Y Position Error')
    axs[1].set_ylabel('Error (m)')
    axs[1].grid(True)
    axs[1].legend()
    
    # 绘制theta误差子图
    for i, N in enumerate(N_values):
        time_steps_N = range(len(results[N]['error_theta']))
        axs[2].plot(time_steps_N, results[N]['error_theta'], 
                   color=colors[i % len(colors)], 
                   linestyle=line_styles[i % len(line_styles)],
                   label=f'N = {N}')
    axs[2].set_title('Theta Orientation Error')
    axs[2].set_xlabel('Time Steps')
    axs[2].set_ylabel('Error (rad)')
    axs[2].grid(True)
    axs[2].legend()
    
    # 调整子图之间的间距
    plt.tight_layout()
    
    # 保存图形
    plt.savefig("offset_imgs/state_errors_comparison.png", bbox_inches='tight')
    
    # 显示图形
    plt.show()
    
    # 绘制代价函数图
    plt.figure(figsize=(10, 6))
    plt.plot(cost_decrease_list, drawstyle='steps', label=r"$V_f(f(x, k_x))-V_f(x, k_x)$")
    plt.plot(minus_stage_cost, drawstyle='steps', label=r"$-l(x, u)$")
    plt.title('Cost Functions')
    plt.legend()
    plt.grid(True)
    plt.savefig("offset_imgs/cost_functions.png", bbox_inches='tight')
    plt.show()
    print(f"Result: {'Success' if success else 'Incomplete'}")
