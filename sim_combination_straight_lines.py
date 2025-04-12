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
import casadi as ca
import pandas as pd
from tqdm import tqdm
import math
import time
from settings import *
import scipy.linalg as la

def is_in_terminal_set(P_matrix, state, radius):
    ref_state = [0, 0, 0]
    error_x = state[0] - ref_state[0]
    error_y = state[1] - ref_state[1]
    error_theta = state[2] - ref_state[2]

    # Normalize heading angle
    while error_theta > np.pi:
        error_theta -= 2 * np.pi
    while error_theta < -np.pi:
        error_theta += 2 * np.pi

    error_vector = np.array([error_x, error_y, error_theta])
    quadratic_form = error_vector.T @ P_matrix @ error_vector

    return float(quadratic_form) <= radius

def load_trajectory_from_csv(csv_file):
    # Load trajectory from CSV with heading information
    df = pd.read_csv(csv_file)
    print(df)
    x = df['X'].values
    y = df['Y'].values
    theta = df['Theta'].values
    print(f"Loaded {len(x)} trajectory points")
    trajectory = [[x[i], y[i], theta[i]] for i in range(len(x))]
    return trajectory, x, y, theta

def CalculateAMatrix(dt, v_ref, theta_ref):
    # Discrete-time state transition matrix A
    A_d = np.eye(3)
    A_d[0, 2] = -dt * v_ref * np.sin(theta_ref)
    A_d[1, 2] = dt * v_ref * np.cos(theta_ref)
    return A_d

def CalculateBMatrix(dt, v_ref, theta_ref, sigma_ref, L):
    # Discrete-time control input matrix B
    B_d = np.zeros((3, 2))
    B_d[0, 0] = dt * np.cos(theta_ref)
    B_d[1, 0] = dt * np.sin(theta_ref)
    B_d[2, 0] = dt * np.tan(sigma_ref) / L
    B_d[2, 1] = dt * v_ref / (L * np.cos(sigma_ref)**2)
    return B_d

def dynamicUpdate(MatrixA, MatrixB, error_k, inputerror_k):
    # Update state error using discrete linear system
    error_k = np.array(error_k).reshape(-1, 1)
    inputerror_k = np.array(inputerror_k).reshape(-1, 1)
    update_error = np.dot(MatrixA, error_k) + np.dot(MatrixB, inputerror_k)
    return update_error

def find_lookahead_point(pos, sampled_trajectory, lookahead_distance):
    # Find a point in front of the vehicle at the lookahead distance
    current_position = pos[:2]
    current_heading = pos[-1]
    heading_vector = np.array([np.cos(current_heading), np.sin(current_heading)])
    min_dist = float('inf')
    result_idx = len(sampled_trajectory) - 1

    for i, point in enumerate(sampled_trajectory):
        dx = point[0] - current_position[0]
        dy = point[1] - current_position[1]
        dist = np.sqrt(dx**2 + dy**2)
        point_vector = np.array([dx, dy])
        if np.linalg.norm(point_vector) > 0:
            point_vector /= np.linalg.norm(point_vector)
        is_in_front = np.dot(heading_vector, point_vector) > 0
        if is_in_front and dist >= lookahead_distance and dist < min_dist:
            min_dist = dist
            result_idx = i

    target_point = sampled_trajectory[result_idx]
    return target_point

def sample_trajectory(original_trajectory, num_segments=5):
    # Sample several key points from the trajectory
    traj_length = len(original_trajectory)
    segment_size = max(1, traj_length // num_segments)
    sampled_indices = [segment_size * i for i in range(1, num_segments + 1) if segment_size * i < traj_length]
    if traj_length - 1 not in sampled_indices:
        sampled_indices.append(traj_length - 1)
    print(sampled_indices)
    return [original_trajectory[i] for i in sampled_indices]

def mpc_controller(pos, reference_traj, sampled_traj, N, dt, L, lookahead_distance=2.0, initial_state=None, v_ref=2, sigma_ref=0):
    # Use initial state if provided
    if initial_state is not None:
        pos = initial_state

    # Find the reference point ahead of the vehicle
    x_ref, y_ref, theta_ref = find_lookahead_point(pos, sampled_traj, lookahead_distance)
    print(f'ref: {x_ref, y_ref, theta_ref}')

    # Create optimization variables
    delta_v = ca.SX.sym('delta_v', N)
    delta_sigma = ca.SX.sym('delta_sigma', N)
    opt_vars = ca.vertcat(ca.reshape(delta_v, -1, 1), ca.reshape(delta_sigma, -1, 1))

    print(f'(x, y, theta) ref: {x_ref, y_ref, theta_ref}')

    # Define weight matrices
    Q = np.diag([1000, 1000, 1000])
    R = np.diag([2, 2])

    A_d = CalculateAMatrix(dt, v_ref, theta_ref)
    B_d = CalculateBMatrix(dt, v_ref, theta_ref, sigma_ref, L)
    P = la.solve_discrete_are(A_d, B_d, Q, R)
    beta = 15

    # Compute initial state error
    e_x = pos[0] - x_ref
    e_y = pos[1] - y_ref
    e_theta = pos[2] - theta_ref

    while e_theta > np.pi:
        e_theta -= 2 * np.pi
    while e_theta < -np.pi:
        e_theta += 2 * np.pi

    cost = 0
    e_k = np.array([e_x, e_y, e_theta])
    final_state_error = None

    # Loop over prediction horizon
    for i in range(N):
        state_cost = e_k @ Q @ e_k
        u_k = np.array([delta_v[i], delta_sigma[i]])
        control_cost = u_k @ R @ u_k
        stage_cost = state_cost + control_cost
        cost += stage_cost
        e_k = A_d @ e_k + B_d @ u_k
        if i == N - 1:
            final_state_error = e_k

    # Add terminal cost
    if final_state_error is not None:
        terminal_cost = beta * final_state_error @ P @ final_state_error
        cost += terminal_cost

    # Define and solve optimization problem
    nlp = {'x': opt_vars, 'f': cost}
    opts = {
        'ipopt.print_level': 0,
        'ipopt.max_iter': 30,
        'ipopt.tol': 1e-4,
        'ipopt.acceptable_tol': 1e-4,
        'print_time': 0
    }

    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

    # Define bounds for optimization variables
    lbx = [-1.0] * N + [-0.4] * N
    ubx = [1.0] * N + [0.4] * N

    sol = solver(x0=np.zeros(2 * N), lbx=lbx, ubx=ubx)
    opt_sol = sol['x'].full().flatten()
    optimal_cost = float(sol['f'])

    optimal_delta_v = opt_sol[0]
    optimal_delta_sigma = opt_sol[N]

    v_optimal = v_ref + optimal_delta_v
    sigma_optimal = sigma_ref + optimal_delta_sigma

    # Compute actual stage cost
    u_actual = np.array([optimal_delta_v, optimal_delta_sigma])
    actual_control_cost = u_actual @ R @ u_actual
    actual_stage_cost = (e_x**2 * Q[0, 0] + e_y**2 * Q[1, 1] + e_theta**2 * Q[2, 2]) + actual_control_cost

    return v_optimal, sigma_optimal, optimal_cost, actual_stage_cost, optimal_delta_v, optimal_delta_sigma, P



def initialize_main_plot():
    """Initialize only the main trajectory plot"""
    # Main trajectory plot
    fig1, ax1 = initialize_plot()
    return fig1, ax1


def run_prius_with_controller(csv_file="bicycle_constrained_trajectory.csv", 
                               v_ref=4,
                               lookahead_distance=5.0, dt=0.1,
                               N=30,
                               custom_start_pos=None):

    # Load full trajectory and sampled control points
    trajectory, ref_x, ref_y, theta_ref = load_trajectory_from_csv(csv_file)
    sampled_traj, ref_x0, ref_y0, theta_ref0 = load_trajectory_from_csv("control_points.csv")
    print(sampled_traj)

    # Set maximum steps
    n_steps = min(100000, len(trajectory) * 10)
    print(f"Maximum steps: {n_steps}")

    radius = 4  # Terminal set radius

    # Set up vehicle model
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

    # Create simulation environment
    env = UrdfEnv(dt=dt, robots=robots, render=False)

    # Set up visualization
    fig1, ax1 = initialize_main_plot()

    # Plot reference trajectory and start/end points
    ax1.plot(ref_x, ref_y, 'g--', linewidth=2, label='Reference Trajectory')
    ax1.plot(ref_x[0], ref_y[0], 'go', markersize=8, label='Trajectory Start')
    ax1.plot(ref_x[-1], ref_y[-1], 'ro', markersize=8, label='Trajectory End')
    ax1.legend()

    # Plot sampled trajectory points
    for ref in sampled_traj:
        ax1.plot(ref[0], ref[1], 'x')

    # Determine initial position
    if custom_start_pos is not None and len(custom_start_pos) >= 3:
        pos0 = np.array([custom_start_pos[0], custom_start_pos[1], custom_start_pos[2], 0.0])
    else:
        print("Warning: Custom start position must be at least [x, y, theta]. Using trajectory start.")
        pos0 = np.array([trajectory[0][0], trajectory[0][1], trajectory[0][2], 0.0])

    print(f"Start: X={pos0[0]:.2f}, Y={pos0[1]:.2f}, Theta={pos0[2]:.2f}")
    print(f"End: X={trajectory[-1][0]:.2f}, Y={trajectory[-1][1]:.2f}")

    # Plot initial position
    ax1.plot(pos0[0], pos0[1], 'mo', markersize=8, label='Vehicle Start')
    ax1.legend()

    # Reset simulation
    ob = env.reset(pos=pos0[0:3])

    # Initialize trajectory tracking
    actual_x = [pos0[0]]
    actual_y = [pos0[1]]

    action = np.array([0.0, 0.0])
    update_interval = max(1, len(trajectory) // 100)

    # Logging
    time_points = []
    velocity_inputs = []
    steering_inputs = []
    error_x_list = []
    error_y_list = []
    error_theta_list = []
    rmse_list = []
    velocity_error_list = []
    steering_error_list = []
    optimal_cost_list = []
    stage_cost_list = []
    cost_decrease_list = []
    negative_stage_cost_list = []

    prev_optimal_cost = None

    with tqdm(total=n_steps, desc="Progress") as progress_bar:
        for i in range(n_steps):
            # Step simulation
            ob, reward, terminated, truncated, info = env.step(action)
            sigma_ref = 0
            pos = ob['robot_0']['joint_state']['position']
            pos = CoG_to_RearAxle(pos)

            actual_x.append(pos[0])
            actual_y.append(pos[1])
            current_time = i * dt
            time_points.append(current_time)

            # Compute control inputs via MPC
            velocity, steering_angle, optimal_cost, stage_cost, velocity_error, steer_error, P = mpc_controller(
                pos, trajectory, sampled_traj, N, dt, 4.6, 
                lookahead_distance=lookahead_distance, 
                v_ref=v_ref, sigma_ref=0
            )

            # Log cost information
            optimal_cost_list.append(optimal_cost)
            stage_cost_list.append(stage_cost)

            if is_in_terminal_set(P, pos, radius): 
                cost_decrease = optimal_cost - prev_optimal_cost
                cost_decrease_list.append(cost_decrease)
                negative_stage_cost_list.append(-stage_cost)

            prev_optimal_cost = optimal_cost

            # Log control inputs
            velocity_inputs.append(velocity)
            steering_inputs.append(steering_angle)

            # Find current reference point and compute tracking error
            ref_pos = find_lookahead_point(pos, trajectory, lookahead_distance)
            error_x = pos[0] - ref_pos[0]
            error_y = pos[1] - ref_pos[1]
            error_theta = pos[2] - ref_pos[2]
            rmse = np.sqrt(error_x**2 + error_y**2)

            # Log input errors
            velocity_error_list.append(velocity_error)
            steering_error_list.append(steer_error)

            # Normalize heading error
            while error_theta > np.pi:
                error_theta -= 2 * np.pi
            while error_theta < -np.pi:
                error_theta += 2 * np.pi

            error_x_list.append(error_x)
            error_y_list.append(error_y)
            error_theta_list.append(error_theta)
            rmse_list.append(rmse)

            # Display cost info periodically
            if i % 1 == 0 and len(cost_decrease_list) > 0:
                print(f"Cost Decrease: {cost_decrease_list[-1]:.4f} Stage Cost: {stage_cost:.4f}")

            # Set action for next step
            action = np.array([velocity, steering_angle])

            # Visualization update
            if i % update_interval == 0:
                update_vehicle(ax1, pos)
                if len(actual_x) > update_interval:
                    ax1.plot(actual_x[-update_interval:], actual_y[-update_interval:], 'b-', linewidth=1.5)
                else:
                    ax1.plot(actual_x, actual_y, 'b-', linewidth=1.5)
                fig1.canvas.draw_idle()
                plt.pause(0.001)

            progress_bar.update(1)

            # Success condition check
            dist_to_end = np.sqrt((pos[0] - trajectory[-1][0])**2 + (pos[1] - trajectory[-1][1])**2)
            if dist_to_end < 0.3:
                print(f'Success! Distance to goal: {dist_to_end:.2f}m')
                ax1.plot(actual_x, actual_y, 'b-', linewidth=1.5, label='Actual Path')
                update_vehicle(ax1, pos)
                ax1.legend()
                fig1.savefig("mpc_trajectory_custom_start.png", bbox_inches='tight')
                plot_inputs_and_errors(time_points, velocity_inputs, steering_inputs, 
                                       error_x_list, error_y_list, error_theta_list)
                env.close()
                return True, rmse_list, error_x_list, error_y_list, error_theta_list, \
                       [velocity_error_list, steering_error_list], cost_decrease_list, \
                       negative_stage_cost_list, actual_x, actual_y, sampled_traj

            elif pos[0] - trajectory[-1][0] >= 0.5:
                env.close()
                return True, rmse_list, error_x_list, error_y_list, error_theta_list, \
                       [velocity_error_list, steering_error_list], cost_decrease_list, \
                       negative_stage_cost_list, actual_x, actual_y, sampled_traj

            if terminated:
                break

    # Final visualization
    ax1.plot(actual_x, actual_y, 'b-', linewidth=1.5, label='Actual Path')
    ax1.legend()
    fig1.savefig("mpc_trajectory_custom_start.png", bbox_inches='tight')

    plot_inputs_and_errors(time_points, velocity_error_list, steering_error_list, 
                           error_x_list, error_y_list, error_theta_list)

    env.close()
    return False, rmse_list, error_x_list, error_y_list, error_theta_list, [velocity_error_list, steering_error_list], cost_decrease_list, negative_stage_cost_list, sampled_traj

def plot_inputs_and_errors(time_points, velocity_inputs, steering_inputs, 
                          error_x_list, error_y_list, error_theta_list):
    """
    Create and save plots for control inputs and errors at the end of simulation
    """
    # Create figure for inputs
    fig2, (ax_vel, ax_steer) = plt.subplots(2, 1, figsize=(10, 8))
    ax_vel.set_title('Velocity Error')
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('Velocity (m/s)')
    ax_vel.set_ylim([-5, 5])
    ax_vel.plot(time_points, velocity_inputs, drawstyle='steps')
    ax_vel.grid(True)
    
    ax_steer.set_title('Steering Angle Input')
    ax_steer.set_xlabel('Time (s)')
    ax_steer.set_ylabel('Steering Angle (rad)')
    ax_steer.plot(time_points, steering_inputs, drawstyle='steps')
    ax_steer.grid(True)
    
    plt.tight_layout()
    fig2.savefig("multi_straight_line_imgs/mpc_control_inputs.png", bbox_inches='tight')
    
    # Create figure for errors
    fig3, (ax_err_x, ax_err_y, ax_err_theta, ax_err_dist) = plt.subplots(4, 1, figsize=(10, 10))
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
    fig3.savefig("multi_straight_line_imgs/mpc_tracking_errors.png", bbox_inches='tight')

if __name__ == "__main__":
    csv_file = "straight_line_trajectory.csv"
    print(f'Running trajectory from: {csv_file}')
    
    # Define initial state
    initial_pos = [-48, -4, 0.4]

    # Define prediction horizons to evaluate
    N_values = [3, 5, 10, 30]

    # Dictionary to store results for each N
    results = {}

    # Run simulation for each prediction horizon
    for N in N_values:
        print(f"Running simulation with N = {N}")

        success, rmse_error_list, error_x_list, error_y_list, error_theta_list, \
        [velocity_error, sigma_error], cost_decrease_list, minus_stage_cost, \
        actual_x, actual_y, sampled_traj = run_prius_with_controller(
            csv_file=csv_file,
            v_ref=4,
            lookahead_distance=0.3,  # Lookahead distance (meters)
            dt=0.05,                 # Simulation time step (seconds)
            N=N,
            custom_start_pos=initial_pos
        )

        # Store simulation results
        results[N] = {
            'rmse': rmse_error_list,
            'error_x': error_x_list,
            'error_y': error_y_list,
            'error_theta': error_theta_list,
            'actual_x': actual_x,
            'actual_y': actual_y
        }

    # === Plot Error Metrics Over Time ===
    plt.figure(figsize=(16, 12))

    # RMSE plot
    plt.subplot(4, 1, 1)
    for N in N_values:
        print(f"N={N}, rmse length: {len(results[N]['rmse'])}")
        time_steps = np.arange(len(results[N]['rmse'])) * 0.05
        plt.plot(time_steps, results[N]['rmse'], label=f'N = {N}')
    plt.title('RMSE (Distance to Reference Point)')
    plt.xlabel('Time (s)')
    plt.ylabel('Distance Error (m)')
    plt.legend()
    plt.grid(True)

    # X error plot
    plt.subplot(4, 1, 2)
    for N in N_values:
        print(f"N={N}, error_x length: {len(results[N]['error_x'])}")
        time_steps = np.arange(len(results[N]['error_x'])) * 0.05
        plt.plot(time_steps, results[N]['error_x'], label=f'N = {N}')
    plt.title('X Position Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error in X (m)')
    plt.legend()
    plt.grid(True)

    # Y error plot
    plt.subplot(4, 1, 3)
    for N in N_values:
        time_steps = np.arange(len(results[N]['error_y'])) * 0.05
        plt.plot(time_steps, results[N]['error_y'], label=f'N = {N}')
    plt.title('Y Position Error')
    plt.xlabel('Time (s)')
    plt.ylabel('Error in Y (m)')
    plt.legend()
    plt.grid(True)

    # Theta (heading) error plot
    plt.subplot(4, 1, 4)
    for N in N_values:
        time_steps = np.arange(len(results[N]['error_theta'])) * 0.05
        plt.plot(time_steps, results[N]['error_theta'], label=f'N = {N}')
    plt.title('Heading Error (Theta)')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta Error (rad)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("multi_straight_line_imgs/mpc_comparison_different_N.png", dpi=300)

    # === Plot Trajectories ===
    trajectory, ref_x, ref_y, _ = load_trajectory_from_csv(csv_file)
    plt.figure(figsize=(10, 8))

    # Plot reference trajectory
    plt.plot(ref_x, ref_y, 'k--', linewidth=2, label='Reference Trajectory')

    # Plot actual trajectories for each N
    for N in N_values:
        x_vals = results[N]['actual_x']
        y_vals = results[N]['actual_y']
        plt.plot(x_vals, y_vals, label=f'N = {N}')

    # Finalize trajectory plot
    plt.title('Trajectory Comparison for Different Prediction Horizons (N)')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.grid(True)

    # Plot sampled reference points
    for i, point in enumerate(sampled_traj):
        if i == 0:
            plt.plot(point[0], point[1], 'x', label='Sampled Ref Point', markersize=14)
        else:
            plt.plot(point[0], point[1], 'x', markersize=14)

    plt.legend()
    plt.savefig("multi_straight_line_imgs/mpc_trajectory_comparison_different_N.png", dpi=300)

