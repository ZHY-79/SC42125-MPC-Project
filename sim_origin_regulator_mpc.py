"""
Constant Speed Prius Trajectory Tracking Simulation.
Use update_vehicle function to render the vehicle
Main features:
1. Provides speed and angle control interface
2. Uses constant speed by default
3. Drawing range fixed at -50 to 50
"""

import numpy as np
from urdfenvs.urdf_common.urdf_env import UrdfEnv
from bicycle_model_self import BicycleModelSelf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, Polygon, Ellipse
import casadi as ca  # Non-linear numerical optimization solver
import pandas as pd
from tqdm import tqdm
import math
import time
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
    """Create a single point trajectory [0,0,0]"""
    trajectory = [[0, 0, 0]]  # Single point trajectory [x, y, theta]
    x = trajectory[0][0]
    y = trajectory[0][1]
    theta = trajectory[0][1]
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
    Update the state error of a discrete-time dynamic system
    
    Parameters:
    MatrixA: Discrete state transition matrix A_d
    MatrixB: Discrete input matrix B_d
    error_k: Current state error [e_x, e_y, e_theta]
    inputerror_k: Current input error [delta_v, delta_sigma]
    
    Returns:
    error_k1: Next state error [e_x, e_y, e_theta]
    """
    # Ensure inputs are numpy arrays
    error_k = np.array(error_k).reshape(-1, 1)       # Convert to column vector
    inputerror_k = np.array(inputerror_k).reshape(-1, 1)  # Convert to column vector
    
    # Calculate next state error based on discrete dynamic equation
    # e(k+1) = A_d * e(k) + B_d * δu(k)
    update_error = np.dot(MatrixA, error_k) + np.dot(MatrixB, inputerror_k)
    return update_error

def mpc_controller(pos, reference_traj, N, dt, L, initial_state=None, v_ref=4, sigma_ref=0, Q=np.diag([1, 1, 1]), R=np.diag([1, 1])):
    # If initial state is provided, use it instead of current state
    if initial_state is not None:
        pos = initial_state
    
    # Use single point target trajectory
    x_ref, y_ref, theta_ref = reference_traj[0]
    # Create optimization variables
    delta_v = ca.SX.sym('delta_v', N)                                          # Velocity deviation symbolic variable
    delta_sigma = ca.SX.sym('delta_sigma', N)                                  # Steering angle deviation symbolic variable
    opt_vars = ca.vertcat(ca.reshape(delta_v, -1, 1), ca.reshape(delta_sigma, -1, 1))  # Build optimization vector
    
    # Define state and control weights
    # Q = np.diag([1, 1, 1])                                               # State error weight matrix [x, y, theta]
    # R = np.diag([1, 1])                                                   # Control deviation weight matrix [v, sigma]
    A_d = CalculateAMatrix(dt, v_ref, theta_ref)                               # State transition matrix
    B_d = CalculateBMatrix(dt, v_ref, theta_ref, sigma_ref, L)                 # Control input matrix
                                                                             
    P = la.solve_discrete_are(A_d, B_d, Q, R)
    K = np.linalg.inv(R + B_d.T @ P @ B_d) @ (B_d.T @ P @ A_d) 
    u_feedback = - K @ pos
    # print(f'P {P}')
    print(f'The max eigenvalue: {np.max(np.linalg.eigvals(P))} the min: {np.min(np.linalg.eigvals(P))}')
    beta = 15          # beta * terminal cost
    
    # Calculate initial state error
    e_x = pos[0] - x_ref                                                       # Calculate x-direction error
    e_y = pos[1] - y_ref                                                       # Calculate y-direction error
    e_theta = pos[2] - theta_ref                                               # Calculate heading angle error
    
    # Normalize angle error to [-pi, pi]
    while e_theta > np.pi: e_theta -= 2 * np.pi                                # Handle angle > π case
    while e_theta < -np.pi: e_theta += 2 * np.pi                               # Handle angle < -π case
    
    cost = 0                                                                   # Initialize total cost
    e_k = np.array([e_x, e_y, e_theta])                                        # Initial state error
    
    # Save final predicted state error for terminal cost
    final_state_error = None

    # Use fixed reference point and reference control throughout prediction horizon
    for i in range(N):
        # Calculate state and control costs
        state_cost = e_k @ Q @ e_k                                             # State error weighted quadratic form
        u_k = np.array([delta_v[i], delta_sigma[i]])                           # Current control deviation
        control_cost = u_k @ R @ u_k                                           # Control deviation weighted quadratic form
        stage_cost = state_cost + control_cost                                 # Current stage cost
        cost = cost + stage_cost                                               # Add each stage cost to total

        # Predict next state error
        e_k = A_d @ e_k + B_d @ u_k                                            # Predict next state error
        
        # Save final predicted state error
        if i == N - 1:
            final_state_error = e_k
    
    # Add terminal cost
    terminal_cost = 0
    if final_state_error is not None:
        terminal_cost = beta * final_state_error @ P @ final_state_error       # Terminal state error weighted quadratic form
        cost = cost + terminal_cost                                            # Add terminal cost to total
    
    # Define and solve optimization problem
    nlp = {'x': opt_vars, 'f': cost}                                           # Define nonlinear programming problem
    
    opts = {
        'ipopt.print_level': 0,                                                # Suppress IPOPT output
        'ipopt.max_iter': 30,                                                  # Maximum iterations
        'ipopt.tol': 1e-4,                                                     # Convergence tolerance
        'ipopt.acceptable_tol': 1e-4,                                          # Acceptable tolerance
        'print_time': 0                                                        # Don't print computation time
    }
    
    solver = ca.nlpsol('solver', 'ipopt', nlp, opts)                           # Create IPOPT solver instance
    
    # Set optimization bounds
    lbx = []                                                                   # Lower bounds list
    ubx = []                                                                   # Upper bounds list
    
    for _ in range(N):
        lbx.extend([-1.0])                                                     # Minimum velocity deviation
        ubx.extend([1.0])                                                      # Maximum velocity deviation
    
    for _ in range(N):
        lbx.extend([-0.5])                                                     # Minimum steering angle deviation
        ubx.extend([0.5])                                                      # Maximum steering angle deviation
    
    # Solve optimization problem and extract results
    optimal_cost = None

    sol = solver(x0=np.zeros(2*N), lbx=lbx, ubx=ubx)                           # Solve optimization problem
    opt_sol = sol['x'].full().flatten()                                        # Extract optimization results
    optimal_cost = float(sol['f'])                                             # Extract optimal cost value
        
    optimal_delta_v = opt_sol[0]                                               # Extract velocity deviation
    optimal_delta_sigma = opt_sol[N]                                           # Extract steering angle deviation
        
    # Apply optimization results to calculate final control inputs
    v_optimal = v_ref + optimal_delta_v                                        # Reference velocity plus deviation
    sigma_optimal = sigma_ref + optimal_delta_sigma                            # Reference steering angle plus deviation
        
    # Calculate actual stage cost (using actual delta_v and delta_sigma)
    u_actual = np.array([optimal_delta_v, optimal_delta_sigma])                # Actual input deviation
    actual_control_cost = u_actual @ R @ u_actual                              # Actual control cost
    actual_stage_cost = (e_x**2 * Q[0,0] + e_y**2 * Q[1,1] + e_theta**2 * Q[2,2]) + actual_control_cost  # Actual stage cost
    
    # Calculate current state terminal cost V_f(x)
    current_error = np.array([e_x, e_y, e_theta])
    current_terminal_cost = beta * current_error @ P @ current_error
    
    # Calculate next state terminal cost V_f(f(x,u))
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
    fig3.savefig("origin_imgs/mpc_tracking_errors_single_point.png", bbox_inches='tight')

def run_prius_with_controller(v_ref=4, dt=0.1, N=30, custom_start_pos=None, Q=np.diag([1, 1, 1]), R=np.diag([1, 1])):
    # Use single point trajectory
    trajectory, ref_x, ref_y, theta_ref = create_single_point_trajectory()
    
    # Set maximum steps
    n_steps = 1000  # Limit maximum steps since we only have one target point
    print(f"Maximum steps: {n_steps}")
    
    # Set up robot
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
    
    # Set up visualization
    fig1, ax1 = initialize_main_plot()
    
    # Draw reference trajectory point
    x_points = [-12,0]  # x-coordinate list
    y_points = [0, 0]  # y-coordinate list

    # Draw line connecting these two points
    
    ax1.plot(ref_x, ref_y, 'go', markersize=8, label='Target Point')
    ax1.legend()
    ax1.plot(x_points, y_points, label="Reference Trajectory")
    ax1.set_ylim([-10, 10])
    # Set starting point
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
    update_interval = 5
    
    # Lists for recording data
    time_points = []
    velocity_inputs = []
    steering_inputs = []
    error_x_list = []
    error_y_list = []
    error_theta_list = []
    
    # Record root mean square position error and input error
    rmse_list = []
    velocity_error_list = []
    steering_error_list = []
    
    # Record terminal cost difference and negative stage cost, to verify V_f(f(x,u))-V_f(x) <= -stage_cost
    terminal_cost_decrease_list = []  # V_f(f(x,u))-V_f(x)
    negative_stage_cost_list = []     # -stage_cost
    
    # Terminal set parameters
    terminal_P = None  # Will be obtained from MPC controller
    terminal_radius = 5
    
    # Record P eigenvalues
    min_eigen_P = np.inf
    max_eigen_P = 0

    with tqdm(total=n_steps, desc="Progress") as progress_bar:
        for i in range(n_steps):
            # Step simulation
            ob, _, terminated, _, _ = env.step(action)
            sigma_ref = 0
            pos = ob['robot_0']['joint_state']['position']
            pos = CoG_to_RearAxle(pos)
            
            # Track actual trajectory
            actual_x.append(pos[0])
            actual_y.append(pos[1])
            
            # Calculate time point
            current_time = i * dt
            time_points.append(current_time)
            
            # Use modified MPC controller, now returns current terminal cost and next step terminal cost
            velocity, steering_angle, optimal_cost, stage_cost, current_terminal_cost, next_terminal_cost, P_matrix = mpc_controller(
                pos, trajectory, N, dt, 4.6, v_ref=v_ref, sigma_ref=0, Q=Q, R=R
            )
            
            # Save P matrix on first iteration
            if i == 0 and terminal_P is None:
                terminal_P = P_matrix
            
            

            eigen_v = np.linalg.eigvals(terminal_P)
            min_eigen_P = min(min_eigen_P, np.min(eigen_v))
            max_eigen_P = max(max_eigen_P, np.max(eigen_v))
            # Calculate terminal cost difference V_f(f(x,u))-V_f(x)
            terminal_cost_decrease = next_terminal_cost - current_terminal_cost
            
            print(is_in_terminal_set(terminal_P, pos, terminal_radius))
            # Calculate negative stage cost
            negative_stage_cost = -stage_cost
            
            
            # Check if in terminal set
            if terminal_P is not None and is_in_terminal_set(terminal_P, pos, radius=terminal_radius):
                print(f"V_f(f(x,u))-V_f(x) = {terminal_cost_decrease:.6f}, -stage_cost = {negative_stage_cost:.6f}")
                print(f"Condition satisfied: {terminal_cost_decrease <= negative_stage_cost}")
                terminal_cost_decrease_list.append(terminal_cost_decrease)
                negative_stage_cost_list.append(negative_stage_cost)
            
            # Record control inputs
            velocity_inputs.append(velocity)
            steering_inputs.append(steering_angle)
            
            # Calculate error - fixed target at origin (0,0,0)
            error_x = pos[0] - trajectory[0][0]
            error_y = pos[1] - trajectory[0][1]
            error_theta = pos[2] - trajectory[0][2]
            
            # Calculate root mean square position error
            rmse = np.sqrt(error_x**2 + error_y**2)
            rmse_list.append(rmse)
            
            # Calculate input error
            velocity_error = velocity - v_ref
            steering_error = steering_angle - sigma_ref
            velocity_error_list.append(velocity_error)
            steering_error_list.append(steering_error)
            
            # Normalize heading angle error to [-pi, pi]
            while error_theta > np.pi:
                error_theta -= 2 * np.pi
            while error_theta < -np.pi:
                error_theta += 2 * np.pi
                
            # Record errors
            error_x_list.append(error_x)
            error_y_list.append(error_y)
            error_theta_list.append(error_theta)
            
            # Set next action
            action = np.array([velocity, steering_angle])
           
            # Update visualization
            if i % update_interval == 0:
                # Use update_vehicle function to draw the vehicle
                update_vehicle(ax1, pos)
                
                # Draw trajectory
                if len(actual_x) > update_interval:
                    ax1.plot(actual_x[-update_interval:], actual_y[-update_interval:], 'b-', linewidth=1.5)
                else:
                    ax1.plot(actual_x, actual_y, 'b-', linewidth=1.5)
                
                # Update main figure
                fig1.canvas.draw_idle()
                plt.pause(0.001)
            
            # Update progress bar
            progress_bar.update(1)
            
            # Check success condition
            dist_to_end = np.sqrt((pos[0]-trajectory[0][0])**2 + (pos[1]-trajectory[0][1])**2)
            print(dist_to_end)
            if dist_to_end < 0.01 or velocity<=0.001: #  and abs(pos[2]-trajectory[0][2]) < 0.1:
                print(f'Success! Distance to goal: {dist_to_end:.2f}m, Angle error: {abs(pos[2]-trajectory[0][2]):.2f}rad')
                ax1.plot(actual_x, actual_y, 'b-', linewidth=1.5, label='Actual Path')
                update_vehicle(ax1, pos)
                ax1.legend()
                
                # Save trajectory figure
                fig1.savefig("origin_imgs/mpc_trajectory_single_point.png", bbox_inches='tight')
                
                # Create and save input figures
                plot_inputs_and_errors(time_points, velocity_inputs, steering_inputs, 
                                      error_x_list, error_y_list, error_theta_list)
                print(f'The max eigenvalue of P is {max_eigen_P}, the min eigenvalue of P is {min_eigen_P}')
                env.close()
                return True, rmse_list, error_x_list, error_y_list, error_theta_list, [velocity_error_list, steering_error_list], terminal_cost_decrease_list, negative_stage_cost_list
            
            if terminated:
                break
    
    # Final visualization
    ax1.plot(actual_x, actual_y, 'b-', linewidth=1.5, label='Actual Path')
    ax1.legend()
    
    # Save trajectory figure
    fig1.savefig("origin_imgs/mpc_trajectory_single_point.png", bbox_inches='tight')
    
    
    # Create and save input figures
    plot_inputs_and_errors(time_points, velocity_inputs, steering_inputs, 
                          error_x_list, error_y_list, error_theta_list)
    
    env.close()
    return False, rmse_list, error_x_list, error_y_list, error_theta_list, [velocity_error_list, steering_error_list], terminal_cost_decrease_list, negative_stage_cost_list


if __name__ == "__main__":
    print('Running simulation to target point [0,0,0] with different prediction horizons N')

    initial_pos = [-15, -4, 0]        # Set initial position
    N_values = [1]       # Test different N values
    gain_Q = 1
    gain_P = 1
    Q = gain_Q * np.diag([5, 5, 5])   # State error weight matrix [x, y, theta]
    R = gain_P * np.diag([3, 3])      # Control input weight matrix
    
    # Store results for different N values
    results = {}
    
    # Run simulation for each N value
    for N in N_values:
        print(f"Running simulation with N = {N}")
        success, state_error, error_x_list, error_y_list, error_theta_list, [velocity_error, sigma_error], cost_decrease_list, minus_stage_cost = run_prius_with_controller(
            v_ref=0.01,
            dt=0.05,                     # Simulation time step (s)
            N=N,
            custom_start_pos=initial_pos,
            Q=Q,
            R=R
        )
        
        # Store results
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
    
    # Create three separate charts, each plotting the error of one state variable
    fig, axs = plt.subplots(3, 1, figsize=(12, 12))
    
    # Set colors and line styles to make graphs easier to distinguish
    colors = ['b', 'r', 'g', 'purple']
    line_styles = ['-', '--', ':', '-.']
    
    # Plot x error subplot
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
    
    # Plot y error subplot
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
    
    # Plot theta error subplot
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
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Save figure
    plt.savefig("origin_imgs/state_errors_comparison.png", bbox_inches='tight')
    
    # Show figure
    plt.show()
    
    # Plot cost function graph
    plt.figure(figsize=(10, 6))
    plt.plot(cost_decrease_list, drawstyle='steps', label=r"$V_f(f(x, k_x))-V_f(x, k_x)$")
    plt.plot(minus_stage_cost, drawstyle='steps', label=r"$-l(x, u)$")
    plt.title('Cost Functions')
    plt.legend()
    plt.grid(True)
    plt.savefig("origin_imgs/cost_functions.png", bbox_inches='tight')
    plt.show()
    print(f"Result: {'Success' if success else 'Incomplete'}")