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
import os

# Ensure compare_Q directory exists
if not os.path.exists("compare_Q"):
    os.makedirs("compare_Q")

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

def load_trajectory_from_csv(csv_file):  # Load trajectory points with heading angle
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
    # Ensure inputs are numpy arrays
    error_k = np.array(error_k).reshape(-1, 1)       # Convert to column vector
    inputerror_k = np.array(inputerror_k).reshape(-1, 1)  # Convert to column vector
    
    # Calculate next state error according to discrete dynamic equation
    # e(k+1) = A_d * e(k) + B_d * δu(k)
    update_error = np.dot(MatrixA, error_k) + np.dot(MatrixB, inputerror_k)
    return update_error

def find_lookahead_point(pos, sampled_trajectory, lookahead_distance):
    current_position = pos[:2]
    current_heading = pos[-1]
    
    # Calculate unit vector in the direction of heading
    heading_vector = np.array([np.cos(current_heading), np.sin(current_heading)])
    
    # Find the closest point that satisfies the lookahead distance
    min_dist = float('inf')
    result_idx = len(sampled_trajectory) - 1  # Default to last point
    
    for i, point in enumerate(sampled_trajectory):
        # Calculate vector from current position to reference point
        dx = point[0] - current_position[0]
        dy = point[1] - current_position[1]
        
        # Calculate distance
        dist = np.sqrt(dx**2 + dy**2)
        
        # Calculate unit vector to reference point
        point_vector = np.array([dx, dy])
        if np.linalg.norm(point_vector) > 0:  # Avoid division by zero
            point_vector = point_vector / np.linalg.norm(point_vector)
        
        # Dot product > 0 means the reference point is in front
        is_in_front = np.dot(heading_vector, point_vector) > 0
        
        # If the point is in front and distance is >= lookahead distance, check if it's closer
        if is_in_front and dist >= lookahead_distance and dist < min_dist:
            min_dist = dist
            result_idx = i
    
    # Return x, y coordinates and angle of the found lookahead point
    target_point = sampled_trajectory[result_idx]
    return target_point  # Directly return target point [x, y, theta]

def sample_trajectory(original_trajectory, num_segments=100):
    traj_length = len(original_trajectory)
    segment_size = max(1, traj_length // num_segments)
    
    # First sampling point starts from segment_size, not from 0 (skip starting point)
    sampled_indices = [segment_size * i for i in range(1, num_segments + 1) if segment_size * i < traj_length]
    
    # Ensure the last point is included
    if traj_length - 1 not in sampled_indices:
        sampled_indices.append(traj_length - 1)
    print(sampled_indices)
    # Return actual trajectory points rather than indices
    return [original_trajectory[i] for i in sampled_indices]

def mpc_controller(pos, reference_traj, sampled_traj, N, dt, L, lookahead_distance=2.0, initial_state=None, v_ref=2, sigma_ref=0, Q_gain=5, R_gain=3):      # Note: vref = 0 here because we want to stop, but it can't be solved

    # If initial state is provided, use it instead of current state
    if initial_state is not None:
        pos = initial_state
    
    x_ref, y_ref, theta_ref = find_lookahead_point(pos, sampled_traj, lookahead_distance)
    
    # Create optimization variables
    delta_v = ca.SX.sym('delta_v', N)                                          # Velocity deviation symbolic variable
    delta_sigma = ca.SX.sym('delta_sigma', N)                                  # Steering angle deviation symbolic variable
    opt_vars = ca.vertcat(ca.reshape(delta_v, -1, 1), ca.reshape(delta_sigma, -1, 1))  # Build optimization vector
    
    print(f'(x, y, theta) ref: {x_ref, y_ref, theta_ref}')
    # Define state and control weights
    Q = np.diag([1, 1, 1])                                               # State error weight matrix [x, y, theta]
    R = np.diag([1, 1])                                                  # Control deviation weight matrix [v, sigma]
    Q = Q * Q_gain
    R = R * R_gain
    A_d = CalculateAMatrix(dt, v_ref, theta_ref)                         # State transition matrix
    B_d = CalculateBMatrix(dt, v_ref, theta_ref, sigma_ref, L)           # Control input matrix
                                                                        
    P = la.solve_discrete_are(A_d, B_d, Q, R)
    beta = 15     # beta * terminal cost
    # 5. Calculate initial state error
    e_x = pos[0] - x_ref                                                 # Calculate x direction error
    e_y = pos[1] - y_ref                                                 # Calculate y direction error
    e_theta = pos[2] - theta_ref                                         # Calculate heading angle error
    
    # Normalize angle error to [-pi, pi]
    while e_theta > np.pi: e_theta -= 2 * np.pi                          # Handle angle > π
    while e_theta < -np.pi: e_theta += 2 * np.pi                         # Handle angle < -π
    
    cost = 0                                                             # Initialize total cost
    e_k = np.array([e_x, e_y, e_theta])                                  # Initial state error
    
    # Save final predicted state error for later terminal cost
    final_state_error = None

    # 6. Use fixed reference point and reference control throughout prediction horizon
    for i in range(N):
        # Calculate state and control costs
        state_cost = e_k @ Q @ e_k                                       # State error weighted quadratic
        u_k = np.array([delta_v[i], delta_sigma[i]])                     # Current control deviation
        control_cost = u_k @ R @ u_k                                     # Control deviation weighted quadratic
        stage_cost = state_cost + control_cost                           # Current stage cost
        cost = cost + stage_cost                                         # Accumulate costs at each stage

        # Predict next state error
        e_k = A_d @ e_k + B_d @ u_k                                      # Predict next state error
        
        # Save final predicted state error
        if i == N - 1:
            final_state_error = e_k
    
    # 7. Add terminal cost
    if final_state_error is not None:
        terminal_cost = beta * final_state_error @ P @ final_state_error        # Terminal state error weighted quadratic
        cost = cost + terminal_cost                                      # Add terminal cost to total cost
    
    # 8. Define and solve optimization problem
    nlp = {'x': opt_vars, 'f': cost}                                     # Define nonlinear programming problem
    
    opts = {
        'ipopt.print_level': 0,                                          # Suppress IPOPT output
        'ipopt.max_iter': 30,                                            # Maximum iterations
        'ipopt.tol': 1e-4,                                               # Convergence tolerance
        'ipopt.acceptable_tol': 1e-4,                                    # Acceptable tolerance
        'print_time': 0                                                  # Don't print computation time
    }
    
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)                     # Create IPOPT solver instance
    
    # 9. Set optimization boundaries
    lbx = []                                                             # Lower bounds list
    ubx = []                                                             # Upper bounds list
    
    for _ in range(N):
        lbx.extend([-1.0])                                               # Minimum velocity deviation
        ubx.extend([1.0])                                                # Maximum velocity deviation
    
    for _ in range(N):
        lbx.extend([-0.5])                                               # Minimum steering angle deviation
        ubx.extend([0.5])                                                # Maximum steering angle deviation
    
    # 10. Solve optimization problem and extract results
    optimal_cost = None

    sol = solver(x0=np.zeros(2*N), lbx=lbx, ubx=ubx)                     # Solve optimization problem
    opt_sol = sol['x'].full().flatten()                                  # Extract optimization results
    optimal_cost = float(sol['f'])                                       # Extract optimal cost value
        
    optimal_delta_v = opt_sol[0]                                         # Extract velocity deviation
    optimal_delta_sigma = opt_sol[N]                                     # Extract steering angle deviation
        
    # Apply optimization results to calculate final control inputs
    v_optimal = v_ref + optimal_delta_v                                  # Reference velocity + deviation
    sigma_optimal = sigma_ref + optimal_delta_sigma                      # Reference steering angle + deviation
        
    # Calculate stage cost for actual inputs used (using actual delta_v and delta_sigma)
    u_actual = np.array([optimal_delta_v, optimal_delta_sigma])          # Actual input deviation
    actual_control_cost = u_actual @ R @ u_actual                        # Actual control cost
    actual_stage_cost = (e_x**2 * Q[0,0] + e_y**2 * Q[1,1] + e_theta**2 * Q[2,2]) + actual_control_cost  # Actual stage cost
    return v_optimal, sigma_optimal, optimal_cost, actual_stage_cost, optimal_delta_v, optimal_delta_sigma, P           # Return optimized control inputs, optimal cost and current stage cost


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

    # Load trajectory
    trajectory, ref_x, ref_y, theta_ref = load_trajectory_from_csv(csv_file)
    # Set maximum steps
    n_steps = min(100000, len(trajectory) * 10)
    print(f"Maximum steps: {n_steps}")
    sampled_traj = sample_trajectory(trajectory)
    radius = 4
    # Setup robot
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

    # Create environment
    env = UrdfEnv(dt=dt, robots=robots, render=False)
    
    # Setup visualization
    fig1, ax1 = initialize_main_plot()
    
    # Plot reference trajectory
    ax1.plot(ref_x, ref_y, 'g--', linewidth=2, label='Reference Trajectory')
    
    # Plot start and end points
    ax1.plot(ref_x[0], ref_y[0], 'go', markersize=8, label='Trajectory Start')
    ax1.plot(ref_x[-1], ref_y[-1], 'ro', markersize=8, label='Trajectory End')
    
    ax1.legend()
    first_point = ax1.plot(sampled_traj[0][0], sampled_traj[0][1], 'x', color='blue', label='Reference Points')
    for ref in sampled_traj:
        ax1.plot(ref[0], ref[1], 'x', color='blue')
    ax1.legend()
        
    # Set starting point
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
    
    # Mark starting point on the map
    ax1.plot(pos0[0], pos0[1], 'mo', markersize=8, label='Vehicle Start')
    ax1.legend()
    
    # Reset simulation
    ob = env.reset(pos=pos0[0:3])
    
    # Track trajectory
    actual_x = [pos0[0]]
    actual_y = [pos0[1]]
    
    action = np.array([0.0, 0.0])
    
    # Visualization update interval
    update_interval = max(1, len(trajectory) // 100)
    
    # Lists for data recording
    time_points = []
    velocity_inputs = []
    steering_inputs = []
    error_x_list = []
    error_y_list = []
    error_theta_list = []
    
    # Record RMS position error and input error
    rmse_list = []
    velocity_error_list = []
    steering_error_list = []
    
    # New: Record optimal cost and stage cost
    optimal_cost_list = []
    stage_cost_list = []
    
    # New: Record cost decrease and negative stage cost
    cost_decrease_list = []
    negative_stage_cost_list = []
    
    # Previous optimal cost
    prev_optimal_cost = None
    with tqdm(total=n_steps, desc="Progress") as progress_bar:
        for i in range(n_steps):
            # Step simulation
            ob, reward, terminated, truncated, info = env.step(action)
            sigma_ref = 0
            pos = ob['robot_0']['joint_state']['position']
            pos = CoG_to_RearAxle(pos)
            
            # Track actual trajectory
            actual_x.append(pos[0])
            actual_y.append(pos[1])
            
            # Calculate time point
            current_time = i * dt
            time_points.append(current_time)
            
            # Use MPC to calculate control input and get optimal cost and stage cost
            velocity, steering_angle, optimal_cost, stage_cost, velocity_error, steer_error, P = mpc_controller(
                pos, trajectory, sampled_traj, N, dt, 4.6, 
                lookahead_distance=lookahead_distance, 
                v_ref=v_ref, 
                sigma_ref=0,
                Q_gain=Q_gain,
                R_gain=R_gain
            )
            
            # Record optimal cost and stage cost
            optimal_cost_list.append(optimal_cost)
            stage_cost_list.append(stage_cost)
            
            # Calculate cost decrease (current optimal cost - previous optimal cost)
            if prev_optimal_cost is not None:
                cost_decrease = optimal_cost - prev_optimal_cost
                cost_decrease_list.append(cost_decrease)
                negative_stage_cost_list.append(-stage_cost)
            
            # Update previous optimal cost
            prev_optimal_cost = optimal_cost
    
            # Record control inputs
            velocity_inputs.append(velocity)
            steering_inputs.append(steering_angle)
            
            # Calculate error - find nearest reference point
            ref_pos = find_lookahead_point(pos, trajectory, lookahead_distance)
            
            # Calculate position and heading errors
            error_x = pos[0] - ref_pos[0]
            error_y = pos[1] - ref_pos[1]
            error_theta = pos[2] - ref_pos[2]
            rmse = np.sqrt(error_x**2 + error_y**2)
                
            # Calculate input error (difference between actual input and reference input)
            velocity_error_list.append(velocity_error)
            steering_error_list.append(steer_error)
            
            # Normalize heading angle error to [-pi, pi]
            while error_theta > np.pi:
                error_theta -= 2 * np.pi
            while error_theta < -np.pi:
                error_theta += 2 * np.pi
                
            # Record errors
            error_x_list.append(error_x)
            error_y_list.append(error_y)
            error_theta_list.append(error_theta)
            rmse_list.append(rmse)
            
            # Display status at regular intervals
            if i % 1 == 0:
                if len(cost_decrease_list) > 0:
                    print(f"Cost Decrease: {cost_decrease_list[-1]:.4f} Stage Cost: {stage_cost:.4f}")
            
            # Set next action
            action = np.array([velocity, steering_angle])
            
            # Update visualization
            if i % update_interval == 0:
                # Use update_vehicle function to draw vehicle
                update_vehicle(ax1, pos)
                
                # Plot trajectory
                if len(actual_x) > update_interval:
                    ax1.plot(actual_x[-update_interval:], actual_y[-update_interval:], 'b-', linewidth=1.5)
                else:
                    ax1.plot(actual_x, actual_y, 'b-', linewidth=1.5)
                
                # Update main figure
                fig1.canvas.draw_idle()
                plt.pause(0.001)
            
            # Update progress bar
            progress_bar.update(1)
            
            # Check success conditions
            dist_to_end = np.sqrt((pos[0]-trajectory[-1][0])**2 + (pos[1]-trajectory[-1][1])**2)
            if dist_to_end < 0.3:
                print(f'Success! Distance to goal: {dist_to_end:.2f}m')
                ax1.plot(actual_x, actual_y, 'b-', linewidth=1.5, label='Actual Path')
                update_vehicle(ax1, pos)
                ax1.legend()
                
                # Save trajectory figure
                fig1.savefig(f"compare_Q/mpc_trajectory_Q{Q_gain}.png", bbox_inches='tight')
                
                # Create and save input figure
                plot_inputs_and_errors(time_points, velocity_inputs, steering_inputs, 
                                      error_x_list, error_y_list, error_theta_list, Q_gain)
                
                
                env.close()
                return True, rmse_list, error_x_list, error_y_list, error_theta_list, [velocity_error_list, steering_error_list], cost_decrease_list, negative_stage_cost_list
            
            elif pos[0] - trajectory[-1][0] >=0.5:
                env.close()
                return True, rmse_list, error_x_list, error_y_list, error_theta_list, [velocity_error_list, steering_error_list], cost_decrease_list, negative_stage_cost_list
            if terminated:
                break
    
    # Final visualization
    ax1.plot(actual_x, actual_y, 'b-', linewidth=1.5, label='Actual Path')
    ax1.legend()
    
    # Save trajectory figure
    fig1.savefig(f"compare_Q/mpc_trajectory_Q{Q_gain}.png", bbox_inches='tight')
    
    # Create and save input figure
    plot_inputs_and_errors(time_points, velocity_error_list, steering_error_list, 
                          error_x_list, error_y_list, error_theta_list, Q_gain)
    
    env.close()
    return False, rmse_list, error_x_list, error_y_list, error_theta_list, [velocity_error_list, steering_error_list], cost_decrease_list, negative_stage_cost_list

def plot_inputs_and_errors(time_points, velocity_inputs, steering_inputs, 
                          error_x_list, error_y_list, error_theta_list, Q_gain=5):
    """
    Create and save plots for control inputs and errors at the end of simulation
    """
    # Create figure for inputs
    fig2, (ax_vel, ax_steer) = plt.subplots(2, 1, figsize=(10, 8))
    ax_vel.set_title(f'Velocity Error (Q_gain={Q_gain})')
    ax_vel.set_xlabel('Time (s)')
    ax_vel.set_ylabel('Velocity (m/s)')
    ax_vel.set_ylim([-5, 5])
    ax_vel.plot(time_points, velocity_inputs, drawstyle='steps')
    ax_vel.grid(True)
    
    ax_steer.set_title(f'Steering Angle Input (Q_gain={Q_gain})')
    ax_steer.set_xlabel('Time (s)')
    ax_steer.set_ylabel('Steering Angle (rad)')
    ax_steer.plot(time_points, steering_inputs, drawstyle='steps')
    ax_steer.grid(True)
    
    plt.tight_layout()
    fig2.savefig(f"compare_Q/mpc_control_inputs_Q{Q_gain}.png", bbox_inches='tight')
    
    # Create figure for errors
    fig3, (ax_err_x, ax_err_y, ax_err_theta, ax_err_dist) = plt.subplots(4, 1, figsize=(10, 10))
    ax_err_x.set_title(f'Position Error X (Q_gain={Q_gain})')
    ax_err_x.set_xlabel('Time (s)')
    ax_err_x.set_ylabel('Error (m)')
    ax_err_x.plot(time_points, error_x_list, drawstyle='steps')
    ax_err_x.grid(True)
    
    ax_err_y.set_title(f'Position Error Y (Q_gain={Q_gain})')
    ax_err_y.set_xlabel('Time (s)')
    ax_err_y.set_ylabel('Error (m)')
    ax_err_y.plot(time_points, error_y_list, drawstyle='steps')
    ax_err_y.grid(True)
    
    ax_err_theta.set_title(f'Heading Error (Q_gain={Q_gain})')
    ax_err_theta.set_xlabel('Time (s)')
    ax_err_theta.set_ylabel('Error (rad)')
    ax_err_theta.plot(time_points, error_theta_list, drawstyle='steps')
    ax_err_theta.grid(True)

    # Calculate RMSE and display in fourth subplot
    rmse_values = [np.sqrt(x**2 + y**2) for x, y in zip(error_x_list, error_y_list)]
    ax_err_dist.set_title(f'Position RMSE (Q_gain={Q_gain})')
    ax_err_dist.set_xlabel('Time (s)')
    ax_err_dist.set_ylabel('RMSE (m)')
    ax_err_dist.plot(time_points, rmse_values, drawstyle='steps')
    ax_err_dist.grid(True)

    plt.tight_layout()
    fig3.savefig(f"compare_Q/mpc_tracking_errors_Q{Q_gain}.png", bbox_inches='tight')

if __name__ == "__main__":
    # csv_file = "bicycle_short_trajectory.csv"
    # csv_file = "smooth_path_trajectory.csv"
    csv_file = "smooth_multi_point_traj.csv"
    print(f'Running trajectory from: {csv_file}')
    
    initial_pos = [-45, -4, 0.9]  # Initial position
    
    # Fixed N=30, define different Q_gain values for testing
    N = 30
    Q_gain_values = [1, 10, 100, 1000]
    R_gain = 100  # Fixed R_gain
    
    # Save results for different Q_gain values
    results = {}
    
    for Q_gain in Q_gain_values:
        print(f"Running simulation with Q_gain = {Q_gain}")
        
        success, rmse_error_list, error_x_list, error_y_list, error_theta_list, [velocity_error, sigma_error], cost_decrease_list, minus_stage_cost = run_prius_with_controller(
            csv_file=csv_file,
            v_ref=4,
            lookahead_distance=0.2,  # Lookahead distance (m)
            dt=0.05,                 # Simulation time step (s)
            N=N,
            custom_start_pos=initial_pos,
            Q_gain=Q_gain,
            R_gain=R_gain
        )
        
        # Save results for this Q_gain value
        results[Q_gain] = {
            'rmse': rmse_error_list,
            'error_x': error_x_list,
            'error_y': error_y_list,
            'error_theta': error_theta_list,
            'velocity_error': velocity_error,
            'steering_error': sigma_error
        }
    
    # Plot comparison results
    plt.figure(figsize=(16, 12))
    
    # Plot RMSE subplot
    plt.subplot(4, 1, 1)
    for Q_gain in Q_gain_values:
        # Create time steps array for each data series
        time_steps = np.arange(len(results[Q_gain]['rmse'])) * 0.05
        plt.plot(time_steps, results[Q_gain]['rmse'], label=f'Q_gain = {Q_gain}')
    plt.title('Position RMSE for Different Q Gain Values')
    plt.xlabel('Time (s)')
    plt.ylabel('RMSE (m)')
    plt.legend()
    plt.grid(True)
    
    # Plot X error subplot
    plt.subplot(4, 1, 2)
    for Q_gain in Q_gain_values:
        time_steps = np.arange(len(results[Q_gain]['error_x'])) * 0.05
        plt.plot(time_steps, results[Q_gain]['error_x'], label=f'Q_gain = {Q_gain}')
    plt.title('X Position Error for Different Q Gain Values')
    plt.xlabel('Time (s)')
    plt.ylabel('X Error (m)')
    plt.legend()
    plt.grid(True)
    
    # Plot Y error subplot
    plt.subplot(4, 1, 3)
    for Q_gain in Q_gain_values:
        time_steps = np.arange(len(results[Q_gain]['error_y'])) * 0.05
        plt.plot(time_steps, results[Q_gain]['error_y'], label=f'Q_gain = {Q_gain}')
    plt.title('Y Position Error for Different Q Gain Values')
    plt.xlabel('Time (s)')
    plt.ylabel('Y Error (m)')
    plt.legend()
    plt.grid(True)
    
    # Plot theta error subplot
    plt.subplot(4, 1, 4)
    for Q_gain in Q_gain_values:
        time_steps = np.arange(len(results[Q_gain]['error_theta'])) * 0.05
        plt.plot(time_steps, results[Q_gain]['error_theta'], label=f'Q_gain = {Q_gain}')
    plt.title('Theta Error for Different Q Gain Values')
    plt.xlabel('Time (s)')
    plt.ylabel('Theta Error (rad)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("compare_Q/mpc_comparison_different_Q_gains.png", dpi=300)
    
    # Plot control input comparison figure
    plt.figure(figsize=(16, 8))
    
    # Plot velocity input subplot
    plt.subplot(2, 1, 1)
    for Q_gain in Q_gain_values:
        time_steps = np.arange(len(results[Q_gain]['velocity_error'])) * 0.05
        plt.plot(time_steps, results[Q_gain]['velocity_error'], label=f'Q_gain = {Q_gain}')
    plt.title('Velocity Error for Different Q Gain Values')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity Error (m/s)')
    plt.legend()
    plt.grid(True)
    
    # Plot steering angle input subplot
    plt.subplot(2, 1, 2)
    for Q_gain in Q_gain_values:
        time_steps = np.arange(len(results[Q_gain]['steering_error'])) * 0.05
        plt.plot(time_steps, results[Q_gain]['steering_error'], label=f'Q_gain = {Q_gain}')
    plt.title('Steering Angle Error for Different Q Gain Values')
    plt.xlabel('Time (s)')
    plt.ylabel('Steering Error (rad)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("compare_Q/mpc_control_inputs_comparison.png", dpi=300)