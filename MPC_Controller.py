"""
This script implements functions to create MPC controllers with kinematic bicycle model.
Also, a demo showing how the MPC work in a parking lot environment is made. 
In this demo, the vehicle will move from start point to goal point.

The main features include:

1. **MPC Model**:
   - Generate MPC Model.
2. **MPC Solver**:
   - Solve the MPC problem using 'ipopt' solver.
3. **Kinematic bicycle model**:
   - A kinematic model that can be used for simulation.

This script is modular and can be integrated into larger systems for simulation, vehicle path following and obstacle avoidance tasks.
"""
import os
import pyomo.environ as pe
import numpy as np
import matplotlib.pyplot as plt
from Obstacle_Process import Generate_Obstacle_Circles, plot_circles
from Create_Environment import Generate_Walls, Generate_Obstacles, Generate_Obstacle_Centers, create_grid_map, Generate_GoalPos, update_vehicle, initialize_plot, update_vehicle_dynobs, Generate_Dynamic_Obstacles
from Simple_FreeSpace import compute_free_space
from RRT_Star import *
from A_Star_Planning import plan_path_through_goals
import pandas as pd
import pickle


def mpc_model(L, u, x, h):
    """
    Define pyomo vehicle model for MPC.

    Parameters:
    - L: Wheelbase of the vehicle.
    - u: Control input ([Speed, Steering Angle]).
    - x: State of the vehicle ([x, y, theta]).
    - h: Time interval of the MPC controller.

    Returns:
    - dx, dy, dtheta: Change of the state
    """
    # pyomo cos and sin function
    cos_theta = pe.cos(x[2])
    sin_theta = pe.sin(x[2])
    tan_phi = pe.tan(u[1])
    
    # Calculate vehicle motion
    dx = u[0] * cos_theta * h
    dy = u[0] * sin_theta * h
    dtheta = u[0] * tan_phi / L * h
    
    return dx, dy, dtheta

def simulate_bicycle_model(L, u, x, h):
    """
    Define vehicle model for simulation.

    Parameters:
    - L: Wheelbase of the vehicle.
    - u: Control input ([Speed, Steering Angle]).
    - x: State of the vehicle ([x, y, theta]).
    - h: Time interval of the MPC controller.

    Returns:
    - x_next: Next state of the vehicle.
    """
    cos_theta = np.cos(x[2])
    sin_theta = np.sin(x[2])
    tan_phi = np.tan(u[1])

    dx = u[0] * cos_theta * h
    dy = u[0] * sin_theta * h
    dtheta = u[0] * tan_phi / L * h

    x_next = [x[0] + dx, x[1] + dy, x[2] + dtheta]
    return x_next

def calculate_tangent_point(circle_center, radius, query_point):
    """
        Calculate the intersection points between the line connecting point P and the center O and the circle.
    
        Parameters:
            circle_center: tuple, coordinates of the circle center (x_o, y_o)
            radius: float, radius of the circle
            query_point: tuple, coordinates of point P (x_p, y_p)
    
        Returns:
            tuple, intersection points on the circle (x_a, y_a)
    """

    x_o, y_o = circle_center
    x_p, y_p = query_point

    # Calculate a vector from O to P
    dx = x_p - x_o
    dy = y_p - y_o
    dist = pe.sqrt(dx**2 + dy**2)

    # Calculate the tangent_point
    x_a = x_o + radius * dx / dist
    y_a = y_o + radius * dy / dist
    return (x_a, y_a)

def generate_circles_from_vehicle(pos0, a, w, h, n_discs):
    """
        Generate the centers and radii of covering circles based on the rear axle position, heading angle, geometric center offset, dimensions, and the number of circles.
    
        Parameters:
            pos0: np.array([x, y, theta]), the vehicle's rear axle center position (x, y) and heading angle theta (in radians).
            a: Distance from the rear axle to the vehicle's geometric center.
            w: Vehicle width.
            h: Vehicle length.
            n_discs: Number of covering circles.
    
        Returns:
            circles: A list of each circle's center (x, y) and radius r, formatted as [(cx1, cy1, r1), ...].
    """
    
    x_rear, y_rear, theta = pos0

    # Calculate geometry center
    x_center = x_rear + a * pe.cos(theta)
    y_center = y_rear + a * pe.sin(theta)

    # Calculate radius
    disc_radius = pe.sqrt((h / n_discs / 2) ** 2 + (w / 2) ** 2)

    # Circle centers are evenly distributed along the longitudinal axis at the geometric center.
    disc_offsets = np.linspace(
        -h / 2 + h / (2 * n_discs),
        h / 2 - h / (2 * n_discs),
        n_discs
    )
    
    circles = []
    for offset in disc_offsets:
        cx_local = offset
        cy_local = 0

        cx_global = x_center + cx_local * pe.cos(theta) - cy_local * pe.sin(theta)
        cy_global = y_center + cx_local * pe.sin(theta) + cy_local * pe.cos(theta)

        circles.append((cx_global, cy_global, disc_radius))
    
    return circles

def mpc_problem(N, L, h, Q, R, P, current_state, terminal_state, Wall_Dict, Obs_Dict, last_input, t, Dynamic_Obs_Trajs):
    """
    Define the MPC (Model Predictive Control) optimization problem for vehicle control.

    Parameters:
    - N: int, Prediction horizon (number of time steps to optimize).
    - L: float, Wheelbase of the vehicle.
    - h: float, Time step of the MPC controller.
    - Q: list of floats, Weighting factors for state variables ([x, y, theta]).
    - R: list of floats, Weighting factors for control inputs ([speed, steering angle]).
    - P: list of floats, Weighting factors for the terminal state ([x, y, theta]).
    - current_state: list of floats, Current state of the vehicle ([x, y, theta]).
    - terminal_state: list of floats, Desired terminal state of the vehicle ([x, y, theta]).
    - Wall_Dict: dict, Dictionary of the walls in the environment.
    - Obs_Dict: dict, Dictionary of obstacles in the environment.
    - last_input: list of floats, Last control input applied to the vehicle ([speed, steering angle]).
    - t: float, Current time.
    - Dynamic_Obs_Trajs: list of dynamic obstacle trajectories over time.

    Returns:
    - model: Pyomo ConcreteModel, Optimization model for MPC with constraints and objective function.
    """
    
    # Create pyomo model
    model = pe.ConcreteModel()
    
    model.N = N
    model.h = h

    # State Variable
    model.x = pe.Var(range(N+1), domain=pe.Reals)
    model.y = pe.Var(range(N+1), domain=pe.Reals)
    model.theta = pe.Var(range(N+1), domain=pe.Reals)

    # Control Variable
    model.u_s = pe.Var(range(N), domain=pe.Reals)
    model.u_phi = pe.Var(range(N), domain=pe.Reals)

    # Initial constrains
    model.init_constraints = pe.ConstraintList()
    model.init_constraints.add(model.x[0] == current_state[0])
    model.init_constraints.add(model.y[0] == current_state[1])
    model.init_constraints.add(model.theta[0] == current_state[2])

    # Derivative constrains
    model.dyn_constraints = pe.ConstraintList()
    model.dyn_constraints.add(-5 <= model.u_s[0]-last_input[0])
    model.dyn_constraints.add(model.u_s[0]-last_input[0] <= 5)
    model.dyn_constraints.add(-0.5 <= model.u_phi[0]-last_input[1])
    model.dyn_constraints.add(model.u_phi[0]-last_input[1] <= 0.5)
    
    # Initialize the trajectory of dynamic obstacles
    dyn_obs_x1, dyn_obs_y1, dyn_obs_r1 = dyn_obs1(t, N+1, h)
    dyn_obs_x2, dyn_obs_y2, dyn_obs_r2 = dyn_obs2(t, N+1, h)
    dyn_obs_x3, dyn_obs_y3, dyn_obs_r3 = dyn_obs3(t, N+1, h)
    dyn_obs_x4, dyn_obs_y4, dyn_obs_r4 = dyn_obs4(t, N+1, h)
    dyn_obs_x5, dyn_obs_y5, dyn_obs_r5 = dyn_obs5(t, N+1, h)
    dyn_obs_x6, dyn_obs_y6, dyn_obs_r6 = dyn_obs6(t, N+1, h)
    dyn_obs_x7, dyn_obs_y7, dyn_obs_r7 = dyn_obs7(t, N+1, h)
    dyn_obs_x8, dyn_obs_y8, dyn_obs_r8 = dyn_obs8(t, N+1, h)
    dyn_obs_x9, dyn_obs_y9, dyn_obs_r9 = dyn_obs9(t, N+1, h)
    dyn_obs_x10, dyn_obs_y10, dyn_obs_r10 = dyn_obs10(t, N+1, h)
    dyn_obs_x11, dyn_obs_y11, dyn_obs_r11 = dyn_obs11(t, N+1, h)
    dyn_obs_x12, dyn_obs_y12, dyn_obs_r12 = dyn_obs12(t, N+1, h)
    dyn_obs_x = [dyn_obs_x1, dyn_obs_x2, dyn_obs_x11, dyn_obs_x12]
    dyn_obs_y = [dyn_obs_y1, dyn_obs_y2, dyn_obs_y11, dyn_obs_y12]
    
    
    for i in range(N):
        dx, dy, dtheta = mpc_model(
            L, [model.u_s[i], model.u_phi[i]],
            [model.x[i], model.y[i], model.theta[i]], h
        )
        # Model Constrains
        model.dyn_constraints.add(model.x[i+1] == model.x[i] + dx)
        model.dyn_constraints.add(model.y[i+1] == model.y[i] + dy)
        model.dyn_constraints.add(model.theta[i+1] == model.theta[i] + dtheta)
        
        # Inputs
        model.dyn_constraints.add(-1 <= model.u_s[i])
        model.dyn_constraints.add(model.u_s[i] <= 6)
        model.dyn_constraints.add(-0.8 <= model.u_phi[i])
        model.dyn_constraints.add(model.u_phi[i] <= 0.8)
    
    # Add Static Obstacles constrains
        '''
    for i in range(N+1):
        a = 1.41
        w = 2.1
        h = 4.76
        pos_0 = [current_state[0], current_state[1], current_state[2]]
        Vehicle_Circles = generate_circles_from_vehicle(pos_0, a, w, h, n_discs=1)
        Circles_Dict = Generate_Obstacle_Circles(Wall_Dict, Obs_Dict)
        for veh_cx, veh_cy, veh_r in Vehicle_Circles:
            for circles in Circles_Dict:
                for obs_cx, obs_cy, obs_r in circles:
                    distance = (veh_cx - obs_cx)**2 + (veh_cy - obs_cy)**2
                    if distance < 25:
                        pos = [model.x[i], model.y[i], model.theta[i]]
                        Vehicle_Circles_MPC = generate_circles_from_vehicle(pos, a, w, h, n_discs=2)
                        for veh_cx_MPC, veh_cy_MPC, veh_r_MPC in Vehicle_Circles_MPC:
                            obs_center = [obs_cx, obs_cy]
                            n = [obs_center[0]-veh_cx, obs_center[1]-veh_cy]
                            veh_center = [veh_cx_MPC, veh_cy_MPC]
                            x_a, y_a = calculate_tangent_point(obs_center, veh_r_MPC+obs_r+0.2, veh_center)
                            
                            model.dyn_constraints.add(n[0]*veh_cx_MPC+ n[1]*veh_cy_MPC <= n[0]*x_a + n[1]*y_a)
    
    for i in range(N+1):
        pos = [model.x[i], model.y[i], model.theta[i]]
        Vehicle_Circles_MPC = generate_circles_from_vehicle(pos, a, w, h, n_discs=2)
        for veh_cx_MPC, veh_cy_MPC, veh_r_MPC in Vehicle_Circles_MPC:
            model.dyn_constraints.add(veh_cy_MPC >= -40+veh_r_MPC+0.2)
            model.dyn_constraints.add(veh_cy_MPC <= 40-veh_r_MPC-0.2)
            model.dyn_constraints.add(veh_cx_MPC >= -25+veh_r_MPC+0.2)
            model.dyn_constraints.add(veh_cx_MPC <= 25-veh_r_MPC-0.2)
            
        if current_state[0]>-15 and current_state[0]<-5:
            if current_state[1]<-30:
                for veh_cx_MPC, veh_cy_MPC, veh_r_MPC in Vehicle_Circles_MPC:
                    model.dyn_constraints.add(veh_cy_MPC <= -30-veh_r_MPC-0.2)
            if current_state[1]<5 and current_state[1]>-5:
                for veh_cx_MPC, veh_cy_MPC, veh_r_MPC in Vehicle_Circles_MPC:
                    model.dyn_constraints.add(veh_cy_MPC <= 5-veh_r_MPC-0.2)  
                    model.dyn_constraints.add(veh_cy_MPC >= -5+veh_r_MPC+0.2)   
            if current_state[1]>30:
                for veh_cx_MPC, veh_cy_MPC, veh_r_MPC in Vehicle_Circles_MPC:
                    model.dyn_constraints.add(veh_cy_MPC >= 30+veh_r_MPC+0.2)
        if current_state[0]>5 and current_state[0]<15:
            if current_state[1]<-30:
                for veh_cx_MPC, veh_cy_MPC, veh_r_MPC in Vehicle_Circles_MPC:
                    model.dyn_constraints.add(veh_cy_MPC <= -30-veh_r_MPC-0.2)
            if current_state[1]<5 and current_state[1]>-5:
                for veh_cx_MPC, veh_cy_MPC, veh_r_MPC in Vehicle_Circles_MPC:
                    model.dyn_constraints.add(veh_cy_MPC <= 5-veh_r_MPC-0.2)  
                    model.dyn_constraints.add(veh_cy_MPC >= -5+veh_r_MPC+0.2)   
            if current_state[1]>30:
                for veh_cx_MPC, veh_cy_MPC, veh_r_MPC in Vehicle_Circles_MPC:
                    model.dyn_constraints.add(veh_cy_MPC >= 30+veh_r_MPC+0.2)
        
        if current_state[1]>5 and current_state[1]<30:
            if current_state[0]<-15:
                for veh_cx_MPC, veh_cy_MPC, veh_r_MPC in Vehicle_Circles_MPC:
                    model.dyn_constraints.add(veh_cx_MPC <= -15-veh_r_MPC-0.2)
            if current_state[0]<5 and current_state[0]>-5:
                for veh_cx_MPC, veh_cy_MPC, veh_r_MPC in Vehicle_Circles_MPC:
                    model.dyn_constraints.add(veh_cx_MPC <= 5-veh_r_MPC-0.2)  
                    model.dyn_constraints.add(veh_cx_MPC >= -5+veh_r_MPC+0.2)   
            if current_state[0]>30:
                for veh_cx_MPC, veh_cy_MPC, veh_r_MPC in Vehicle_Circles_MPC:
                    model.dyn_constraints.add(veh_cx_MPC >= 15+veh_r_MPC+0.2)
        if current_state[1]>-30 and current_state[1]<-5:
            if current_state[0]<-15:
                for veh_cx_MPC, veh_cy_MPC, veh_r_MPC in Vehicle_Circles_MPC:
                    model.dyn_constraints.add(veh_cx_MPC <= -15-veh_r_MPC-0.2)
            if current_state[0]<5 and current_state[0]>-5:
                for veh_cx_MPC, veh_cy_MPC, veh_r_MPC in Vehicle_Circles_MPC:
                    model.dyn_constraints.add(veh_cx_MPC <= 5-veh_r_MPC-0.2)  
                    model.dyn_constraints.add(veh_cx_MPC >= -5+veh_r_MPC+0.2)   
            if current_state[0]>30:
                for veh_cx_MPC, veh_cy_MPC, veh_r_MPC in Vehicle_Circles_MPC:
                    model.dyn_constraints.add(veh_cx_MPC >= 15+veh_r_MPC+0.2)
    '''
    # Add Dynamic Obstacles constrains
    for i in range(N+1):
        a = 1.41
        w = 2.1
        h = 4.76
        pos_0 = [current_state[0], current_state[1], current_state[2]]
        Vehicle_Circles = generate_circles_from_vehicle(pos_0, a, w, h, n_discs=1)
        for veh_cx, veh_cy, veh_r in Vehicle_Circles:
            for j in range(len(dyn_obs_x)):
                distance = (veh_cx-dyn_obs_x[j][0])**2 + (veh_cy-dyn_obs_y[j][0])**2
                # Only account for obstacle within certain range to reduce calculation time
                if distance < 100:
                    pos = [model.x[i], model.y[i], model.theta[i]]
                    Vehicle_Circles_MPC = generate_circles_from_vehicle(pos, a, w, h, n_discs=2)
                    for veh_cx_MPC, veh_cy_MPC, veh_r_MPC in Vehicle_Circles_MPC:
                        dyn_obs_center = [dyn_obs_x[j][i], dyn_obs_y[j][i]]
                        n = [dyn_obs_center[0]-veh_cx, dyn_obs_center[1]-veh_cy]
                        veh_center = [veh_cx, veh_cy]
                        x_a, y_a = calculate_tangent_point(dyn_obs_center, veh_r_MPC+2, veh_center)
                        # Linear constrains
                        model.dyn_constraints.add(n[0]*veh_cx_MPC+ n[1]*veh_cy_MPC <= n[0]*x_a + n[1]*y_a)

    # Define Cost function
    def cost_function(model):
        # State
        cost = sum(
            Q[0] * (model.x[i] - terminal_state[0])**2 + Q[1] * (model.y[i] - terminal_state[1])**2 + Q[2] * (model.theta[i] - terminal_state[2])**2 +
            R[0] * model.u_s[i]**2 + R[1] * model.u_phi[i]**2
            for i in range(N)
        )
        cost += P[0] * (model.x[N] - terminal_state[0])**2 + P[1] * (model.y[N] - terminal_state[1])**2 + P[2] * (model.theta[N] - terminal_state[2])**2
        # Control inputs
        cost += sum(
            1e-1 * (model.u_s[i] - model.u_s[i-1])**2 +
            1e-4 * (model.u_phi[i] - model.u_phi[i-1])**2
            for i in range(1, N)
        )
        return cost
    
    # Set Model
    model.obj = pe.Objective(rule=cost_function, sense=pe.minimize)

    return model

def solve_mpc(N, L, h, Q, R, P, current_state, terminal_state, Wall_Dict, Obs_Dict, last_input, t, Dynamic_Obs_Trajs):
    """
    Solve MPC Problem.

    Parameters:
    - N: int, Prediction horizon (number of time steps to optimize).
    - L: float, Wheelbase of the vehicle.
    - h: float, Time step of the MPC controller.
    - Q: list of floats, Weighting factors for state variables ([x, y, theta]).
    - R: list of floats, Weighting factors for control inputs ([speed, steering angle]).
    - P: list of floats, Weighting factors for the terminal state ([x, y, theta]).
    - current_state: list of floats, Current state of the vehicle ([x, y, theta]).
    - terminal_state: list of floats, Desired terminal state of the vehicle ([x, y, theta]).
    - Wall_Dict: dict, Dictionary of the walls in the environment.
    - Obs_Dict: dict, Dictionary of obstacles in the environment.
    - last_input: list of floats, Last control input applied to the vehicle ([speed, steering angle]).
    - t: float, Current time.
    - Dynamic_Obs_Trajs: list of dynamic obstacle trajectories over time.

    Returns:
    - u_s_opt[0]: Speed of the vehicle.
    - u_phi_opt[0]: Steering angle of the vehicle.
    - x_opt, y_opt: Trajectory calculated by MPC.
    """
    
    model = mpc_problem(N, L, h, Q, R, P, current_state, terminal_state, Wall_Dict, Obs_Dict, last_input, t, Dynamic_Obs_Trajs)

    # Set solver details
    solver = pe.SolverFactory('ipopt')
    solver.options['tol'] = 1e-6
    solver.options['acceptable_tol'] = 1e-4
    solver.options['max_iter'] = 1000
    solver.options['bound_relax_factor'] = 1e-6
    solver.options['constr_viol_tol'] = 1e-6
    solver.options['compl_inf_tol'] = 1e-6
    solver.options['print_level'] = 0
    solver.options['file_print_level'] = 0

    # Get result from solver
    results = solver.solve(model)

    u_s_opt = np.array([model.u_s[i].value for i in range(N)])
    u_phi_opt = np.array([model.u_phi[i].value for i in range(N)])
    x_opt = np.array([model.x[i].value for i in range(N+1)])
    y_opt = np.array([model.y[i].value for i in range(N+1)])
    
    return u_s_opt[0], u_phi_opt[0], x_opt, y_opt


def find_reference_point_and_heading(path_astar, current_point, lookahead_distance, terminal_state):
    """
        Select a reference point from the path based on the current position and calculate the heading angle.
    
        Parameters:
        path_astar : list of np.array
            List of path points, where each element is np.array([x, y]).
        current_point : np.array
            Current point coordinates, shape [x, y, theta].
        lookahead_distance : float
            The required lookahead distance from the current point to select the reference point.
    
        Returns:
        reference_point : np.array
            Coordinates of the reference point [x, y].
        heading : float
            Heading angle of the reference point, in radians.
    """

    # Calculate distance from current state to the path
    distances = [np.linalg.norm(point - current_point[:2]) for point in path_astar]
    
    # Find the point with smallest distance
    closest_index = np.argmin(distances)
    
    # Search reference point accoring to the lookahead distance
    for i in range(closest_index, len(path_astar)):
        if np.linalg.norm(path_astar[i] - current_point[:2]) >= lookahead_distance:
            reference_point = path_astar[i]
            
            # Calculate reference orientation
            if i + 1 < len(path_astar):
                next_point = path_astar[i + 1]
            # Last Point
            else:
                next_point = path_astar[i - 1]
            
            heading = np.arctan2(next_point[1] - reference_point[1], next_point[0] - reference_point[0])
            return [reference_point[0], reference_point[1], heading]
    
    # return last point if not found
    reference_point = path_astar[-1]
    if len(path_astar) > 1:
        heading = np.arctan2(path_astar[-1][1] - path_astar[-2][1], path_astar[-1][0] - path_astar[-2][0])
    else:
        # Default value
        heading = 0.0
        
    return terminal_state

''' !!! Generate Dynamic Obstacles according to the Environment !!! '''
def dyn_obs1(t, N, h):
    dyn_obs_x = []
    dyn_obs_y = []
    dyn_obs_r = []
    for i in range(N):
        dyn_obs_x.append(4 * np.sin(0.5 * (t+h*i) + np.pi / 6))
        dyn_obs_y.append(-5.0)
        dyn_obs_r.append(0.7)
    return dyn_obs_x, dyn_obs_y, dyn_obs_r

def dyn_obs2(t, N, h):
    dyn_obs_x = []
    dyn_obs_y = []
    dyn_obs_r = []
    for i in range(N):
        dyn_obs_x.append(4 * np.sin(0.5 * (t+h*i)))
        dyn_obs_y.append(5.0)
        dyn_obs_r.append(0.7)
    return dyn_obs_x, dyn_obs_y, dyn_obs_r

def dyn_obs3(t, N, h):
    dyn_obs_x = []
    dyn_obs_y = []
    dyn_obs_r = []
    for i in range(N):
        dyn_obs_x.append(4.5)
        dyn_obs_y.append(4 * np.sin(0.5 * (t+h*i) + np.pi / 6))
        dyn_obs_r.append(0.7)
    return dyn_obs_x, dyn_obs_y, dyn_obs_r

def dyn_obs4(t, N, h):
    dyn_obs_x = []
    dyn_obs_y = []
    dyn_obs_r = []
    for i in range(N):
        dyn_obs_x.append(-4.5)
        dyn_obs_y.append(4 * np.sin(0.5 * (t+h*i) + np.pi / 4))
        dyn_obs_r.append(0.7)
    return dyn_obs_x, dyn_obs_y, dyn_obs_r

def dyn_obs5(t, N, h):
    dyn_obs_x = []
    dyn_obs_y = []
    dyn_obs_r = []
    for i in range(N):
        dyn_obs_x.append(4 * np.sin(0.5 * (t+h*i))-20)
        dyn_obs_y.append(5.0)
        dyn_obs_r.append(0.7)
    return dyn_obs_x, dyn_obs_y, dyn_obs_r

def dyn_obs6(t, N, h):
    dyn_obs_x = []
    dyn_obs_y = []
    dyn_obs_r = []
    for i in range(N):
        dyn_obs_x.append(4 * np.sin(0.5 * (t+h*i) + np.pi / 6)-20)
        dyn_obs_y.append(-5.0)
        dyn_obs_r.append(0.7)
    return dyn_obs_x, dyn_obs_y, dyn_obs_r

def dyn_obs7(t, N, h):
    dyn_obs_x = []
    dyn_obs_y = []
    dyn_obs_r = []
    for i in range(N):
        dyn_obs_x.append(4 * np.sin(0.5 * (t+h*i))+20)
        dyn_obs_y.append(5.0)
        dyn_obs_r.append(0.7)
    return dyn_obs_x, dyn_obs_y, dyn_obs_r

def dyn_obs8(t, N, h):
    dyn_obs_x = []
    dyn_obs_y = []
    dyn_obs_r = []
    for i in range(N):
        dyn_obs_x.append(4 * np.sin(0.5 * (t+h*i) + np.pi / 6)+20)
        dyn_obs_y.append(-5.0)
        dyn_obs_r.append(0.7)
    return dyn_obs_x, dyn_obs_y, dyn_obs_r

def dyn_obs9(t, N, h):
    dyn_obs_x = []
    dyn_obs_y = []
    dyn_obs_r = []
    for i in range(N):
        dyn_obs_x.append(4.5)
        dyn_obs_y.append(4 * np.sin(0.5 * (t+h*i))+35)
        dyn_obs_r.append(0.7)
    return dyn_obs_x, dyn_obs_y, dyn_obs_r

def dyn_obs10(t, N, h):
    dyn_obs_x = []
    dyn_obs_y = []
    dyn_obs_r = []
    for i in range(N):
        dyn_obs_x.append(-4.5)
        dyn_obs_y.append(4 * np.sin(0.5 * (t+h*i) + np.pi / 4)+35)
        dyn_obs_r.append(0.7)
    return dyn_obs_x, dyn_obs_y, dyn_obs_r

def dyn_obs11(t, N, h):
    dyn_obs_x = []
    dyn_obs_y = []
    dyn_obs_r = []
    for i in range(N):
        dyn_obs_x.append(4.5)
        dyn_obs_y.append(4 * np.sin(0.5 * (t+h*i))-35)
        dyn_obs_r.append(0.7)
    return dyn_obs_x, dyn_obs_y, dyn_obs_r

def dyn_obs12(t, N, h):
    dyn_obs_x = []
    dyn_obs_y = []
    dyn_obs_r = []
    for i in range(N):
        dyn_obs_x.append(-8)
        dyn_obs_y.append(4 * np.sin(0.5 * (t+h*i))-35)
        dyn_obs_r.append(0.7)
    return dyn_obs_x, dyn_obs_y, dyn_obs_r

def Run_MPC(current_state, terminal_state, path, lookahead_distance, step, Wall_Dict, Obs_Dict, Dynamic_Obs_Traj, last_input, print_log=True):
    """
    Solve one MPC Problem.

    Parameters:
    - current_state: list of floats, Current state of the vehicle ([x, y, theta]).
    - terminal_state: list of floats, Desired terminal state of the vehicle ([x, y, theta]).
    - path: list of floats, A leading path from start to goal.
    - Wall_Dict: dict, Dictionary of the walls in the environment.
    - Obs_Dict: dict, Dictionary of obstacles in the environment.
    - last_input: list of floats, Last control input applied to the vehicle ([speed, steering angle]).
    - print_log: Print debug information.

    Returns:
    - u_s_opt[0]: Speed of the vehicle.
    - u_phi_opt[0]: Steering angle of the vehicle.
    - planned_pathx, planned_pathy: Trajectory calculated by MPC.
    """
    
    L = 2.86  # Wheelbase
    N = 50   # Horizon
    h = 0.01  # time step
    Q = [10, 10, 1]  # State cost
    R = [0, 5]   # Control cost
    P = [100, 100, 10]  # Terminal cost
    
    ref_state = find_reference_point_and_heading(path, current_state[:2], lookahead_distance, terminal_state)
    if print_log:
        print('Reference State: ', ref_state)
    # Solve MPC
    t = h * step  # current time
    u_s, u_phi, planned_pathx, planned_pathy = solve_mpc(N, L, h, Q, R, P, current_state, ref_state, Wall_Dict, Obs_Dict, last_input, t, Dynamic_Obs_Traj)  
    if print_log:
        print('MPC Control: ', u_s, u_phi)
    return u_s, u_phi, planned_pathx, planned_pathy

# Test Main Function
def main():
    # Set MPC variables
    L = 2.86
    N = 50
    h = 0.01
    Q = [10, 10, 1]
    R = [0, 5]
    P = [100, 100, 10]
    
    # Set start and goal points
    current_state = [-20, -35, 0]
    terminal_state = [20, 25, np.pi/2]
    pos0 = np.array([-20, -35, np.pi/4, 0.0])
    goals = np.array([[terminal_state[0],terminal_state[1],terminal_state[2],0.0]])
    goal = goals[0]
    
    # Set max steps
    max_steps = 1500
    # Set stop tolerance
    tolerance = 0.2
    
    # Map
    length = 50
    width = 80
    Walls, Wall_Dict = Generate_Walls(length, width, Type='Parking Lot')

    Obs_centers = Generate_Obstacle_Centers(length, width, n_obstacles=16, Type='Parking Lot')
    Obstacles , Obs_Dict = Generate_Obstacles(Obs_centers, Type='Parking Lot')
    
    grid_size = 0.1
    grid_map, min_x, max_x, min_y, max_y = create_grid_map(Wall_Dict, Obs_Dict, grid_size=grid_size)  
    
    freespace_map = compute_free_space(
        grid_map, grid_size, 2.34+1.45, 2.42-1.45, 1.05, min_x, max_x, min_y, max_y, num_theta=16)
    fig, ax = initialize_plot(grid_map, min_x, max_x, min_y, max_y)
    
    '''
    path_astar = plan_path_through_goals(np.array(current_state), goals, freespace_map, min_x, max_x, min_y, max_y, grid_size, ax, visualize=False)
    
    margin = 2.5
    # Integrate boundary and obstacles
    obstacles_list = []
    for obs in Obs_Dict:
        obstacles_list.append(obs)
    for walls in Wall_Dict:
        obstacles_list.append(walls)
    global_map = {'l': length, 'w': width, 'm': margin, 'o': obstacles_list}
    
    path, path_x, path_y, found_path_x, found_path_y, found_path_yaw, path_length, found_path_curvature = rrt_star(global_map, path_astar,
                                                                                                    pos0, goal)
    path_rear = []
    found_path_rear = []
    found_path_x_rear = []
    found_path_y_rear = []
    found_path_curvature_rear = []
    found_path_yaw_rear = []
    dis = 1.45
    for w in range(len(found_path_x)):
        found_path_rear.append((found_path_x[w] - dis * math.cos(found_path_yaw[w]), 
                               found_path_y[w] - dis * math.sin(found_path_yaw[w]), 
                               found_path_yaw[w], 
                               found_path_curvature[w]))
        found_path_x_rear.append(found_path_x[w] - dis * math.cos(found_path_yaw[w]))
        found_path_y_rear.append(found_path_y[w] - dis * math.sin(found_path_yaw[w]))

    for w in range(1, len(path)):
        path_rear.append((path[w][0] - dis * math.cos(path[w][2]), 
                          path[w][1] - dis * math.sin(path[w][2])))

    
    if path:
        print("Found path:")
        if True:
            draw_path_rrt(ax, path_rear, path_x, path_y, found_path_x_rear, found_path_y_rear)
            plt.show()
            # ax.figure.savefig("trajectory"+str(time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()))+".png")
    else:
        print("No path found.")
        
    #print(path_astar)
    '''
    # Run From pkl file
    df = pd.read_pickle("2025-01-09_13_32_50.pkl")
    found_path_x = df['found_x'].tolist()  
    found_path_y = df['found_y'].tolist()  
    
    path_lead = []
    for x,y in zip(found_path_x,found_path_y):
        path_lead.append([x,y])
    path_lead = np.array(path_lead)
    path_lead = path_lead[~np.isnan(path_lead).any(axis=1)]

    ax.plot(
        found_path_x, 
        found_path_y, 
        '-o', 
        label='RRT Path', 
        color='r', 
        linewidth=0.2,
        alpha=0.5,
        markersize=4
    )
    plt.legend()
    
    # Generate Dynamic_Obs
    Dynamic_Obs_Trajs = []
    Dynamic_Obstacles, Dynamic_Obs_Dicts = Generate_Dynamic_Obstacles(max_steps, n_obstacles=12, dt=0.01, Type="Parking Lot")
    for Dynamic_Obs in Dynamic_Obs_Dicts:
        Dynamic_Obs_Traj = []
        for x,y,z in Dynamic_Obs['geometry']['trajectory']['controlPoints']:
            Dynamic_Obs_Traj.append([x,y])
        Dynamic_Obs_Trajs.append(Dynamic_Obs_Traj)

    trajectory = [current_state]
    V = [0]
    StrAng = [0]

    # Save result data
    save_folder = "saved_trajectories_RRT_MPC"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    saved_data = {
        "current_states": [],
        "dynamic_obstacles": [],
        "planned_paths_x": [],
        "planned_paths_y": []
    }
    
    # Begin Simulation
    for step in range(max_steps):
        dyn_obs_center = []
        for dyn_obs_traj in Dynamic_Obs_Trajs:
            dyn_obs_center.append(dyn_obs_traj[step])

        ref_state = find_reference_point_and_heading(path_lead, current_state[:2], 5, terminal_state)
        print(ref_state)
        
        # Solve MPC 
        t = h * step  # current time
        last_input = np.array([V[-1], StrAng[-1]])

        for Obj in Dynamic_Obs_Trajs:
            Dynamic_Obs_Traj.append(Obj[step])
        u_s, u_phi, planned_pathx, planned_pathy = solve_mpc(N, L, h, Q, R, P, current_state, ref_state, Wall_Dict, Obs_Dict, last_input, t, Dynamic_Obs_Traj)
        
        V.append(u_s)
        StrAng.append(u_phi)
        print(u_s, u_phi)
        
        # Update vehicle position every 20 steps
        if step % 20 == 0:
            update_vehicle_dynobs(ax, current_state, dyn_obs_center, planned_pathx, planned_pathy)
        
        # Save data
        saved_data["current_states"].append(current_state)
        saved_data["dynamic_obstacles"].append(dyn_obs_center)
        saved_data["planned_paths_x"].append(planned_pathx)
        saved_data["planned_paths_y"].append(planned_pathy)

        # Update State
        current_state = simulate_bicycle_model(L, [u_s, u_phi], current_state, 0.01)
        trajectory.append(current_state)
        print(current_state)

        if step % 20 == 0:  # Save every 20 steps
            save_path = os.path.join(save_folder, f"trajectory_step_{step}.png")
            ax.figure.savefig(save_path, dpi=300)
            
        # Check whether the goal state is reached
        if np.linalg.norm([current_state[0] - terminal_state[0], current_state[1] - terminal_state[1]]) < tolerance:
            print(f"Goal reached in {step+1} steps.")
            break
    else:
        print("Maximum steps reached without reaching the goal.")

    # Save data to file
    with open("saved_data.pkl", "wb") as f:
        pickle.dump(saved_data, f)

# Main function
if __name__ == "__main__":
    main()

