import numpy as np
import scipy.linalg as la
import sys
import math
import casadi as ca
from casadi import *
from systems.kinetic_bicycle import *


def get_cost_derivation(
    ctrl_U,
    dX,
    matrix_Q,
    matrix_R,
    num_horizon,
    xvar,
    obstacle
):
    # define control cost derivation
    l_u = np.zeros((2, num_horizon))
    l_uu = np.zeros((2, 2, num_horizon))
    l_x = np.zeros((4, num_horizon))
    l_xx = np.zeros((4, 4, num_horizon))
    # obstacle avoidance
    safety_margin = 0.5
    q1 = 2.25
    q2 = 2.25
    # parameters of the obstacle
    xposition = obstacle.x
    yposition = obstacle.y
    a = obstacle.width
    b = obstacle.height
    for i in range(num_horizon):
        b_dot_ctrl, b_ddot_ctrl = add_control_constraint(ctrl_U[:, i])
        l_u_i = (2 * matrix_R @ ctrl_U[:, i]).reshape(2, 1) + b_dot_ctrl
        l_uu_i = 2 * matrix_R + b_ddot_ctrl
        l_x_i = (2 * matrix_Q @ dX[:, i]).reshape(4, 1) 
        l_xx_i = 2 * matrix_Q 
        # calculate control barrier functions for each obstacle at timestep
        degree = 2
        diffz = (
            xvar[0, i]
            -  xposition
        )
        diffy = xvar[1, i] - yposition
        matrix_P1 = np.diag(
            [1 / (a ** degree), 1 / (b ** degree),0,0]
        )
        diff = np.array([diffz, diffy, 0, 0]).reshape(-1, 1)
        # h = 1 + safety_margin - diff.T @ matrix_P1 @ diff
        # h_dot = -2 * matrix_P1 @ diff
        # _, b_dot_obs, b_ddot_obs = repelling_cost_function(q1, q2, h, h_dot)
        # _, b_dot_obs, b_ddot_obs = repelling_cost_function(q1, q2, h, h_dot)
        # l_x_i += b_dot_obs
        # l_xx_i += b_ddot_obs
        l_u[:, i] = l_u_i.squeeze()
        l_uu[:, :, i] = l_uu_i.squeeze()
        l_xx[:, :, i] = l_xx_i.squeeze()
        l_x[:, i] = l_x_i.squeeze()
    return l_u, l_uu, l_x, l_xx


def repelling_cost_function(q1, q2, c, c_dot):
    b = q1 * np.exp(q2 * c)
    b_dot = q1 * q2 * np.exp(q2 * c) * c_dot
    b_ddot = q1 * (q2 ** 2) * np.exp(q2 * c) * np.matmul(c_dot, c_dot.T)
    return b, b_dot, b_ddot


def select_points(ss_xcurv, Qfun, iter, x0, num_ss_points):
    index_selected = []
    Qfun_vec = Qfun[iter]
    xcurv = ss_xcurv[iter]
    xcurv = np.array(xcurv)
    one_vec = np.ones((xcurv.shape[1],1))
    x0_vec = (np.dot(np.array([x0]).T, one_vec.T)).T
    xcurv = xcurv.T
    diff = xcurv - x0_vec
    norm = la.norm(diff, 1, axis=1)
    idxMinNorm = np.argsort(norm).astype(int)
    print('idx selected', idxMinNorm[0:int(num_ss_points)])

    ss_points = xcurv[idxMinNorm[0:int(num_ss_points)]].T
    sel_Qfun = Qfun_vec[idxMinNorm[0:int(num_ss_points)]]
    Qfun_temp = sel_Qfun.tolist()
    min_index = Qfun_temp.index(min(Qfun_temp))
    return ss_points, sel_Qfun, idxMinNorm[0:int(num_ss_points)]

def add_state_constraint(x):
    matrix_P1 = np.array([[0], [0], [0], [1]])
    q1 = 1.0
    q2 = 1.0
    # Add state barrier max
    c = np.matmul(x.T, matrix_P1) - 3.14/2
    _, b_dot_1, b_ddot_1 = repelling_cost_function(q1, q2, c, matrix_P1)
    # Add state barrier min
    c = -3.14/2 - np.matmul(x.T, matrix_P1)
    _, b_dot_2, b_ddot_2 = repelling_cost_function(q1, q2, c, -matrix_P1)
    b_dot_state = b_dot_1 + b_dot_2
    b_ddot_state = b_ddot_1 + b_ddot_2
    return b_dot_state, b_ddot_state


def add_control_constraint(u):
    # Add constraints of the control input:
    # Accelaration:
    # amin = -2, amax = 2
    # amin < u[0] < amax
    # Yaw rate:
    # theta_dot_min = -pi/2, theta_dot_max = pi/2
    # theta_dot_min < u[1] < theta_dot_max
    matrix_P1 = np.array([[1], [0]])
    matrix_P2 = np.array([[0], [1]])
    q1 = 1.0 # 1.25
    q2 = 1.0
    v_dot_max = 2.0
    v_dot_min = -2.0
    theta_dot_max = 1.57
    theta_dot_min = -1.57
    # Add acceleration barrier max
    c = np.matmul(u.T, matrix_P1) - v_dot_max
    _, b_dot_1, b_ddot_1 = repelling_cost_function(q1, q2, c, matrix_P1)
    # Add acceleration barrier min
    c = v_dot_min - np.matmul(u.T, matrix_P1)
    _, b_dot_2, b_ddot_2 = repelling_cost_function(q1, q2, c, -matrix_P1)
	# Yawrate Barrier Max
    c = (np.matmul(u.T, matrix_P2)) - theta_dot_max
    b_3, b_dot_3, b_ddot_3 = repelling_cost_function(q1, q2, c, matrix_P2)
	# Yawrate Barrier Min
    c = theta_dot_min - np.matmul(u.T, matrix_P2)
    b_4, b_dot_4, b_ddot_4 = repelling_cost_function(q1, q2, c, -matrix_P2)
    b_dot_ctrl =  b_dot_1 + b_dot_2 + b_dot_3 + b_dot_4
    b_ddot_ctrl = b_ddot_1 + b_ddot_2 + b_ddot_3 + b_ddot_4
    return b_dot_ctrl, b_ddot_ctrl


def get_cost_final(
    x,
    xTerminal,
    matrix_Qlambd
):
    # define variables
    l_x = np.zeros((4))
    l_xx = np.zeros((4, 4))
    # Add convex hull constraint as the terminal cost
    # Add state barrier max
    diff = x[:, -1] - xTerminal
    l_x_f = (2 * matrix_Qlambd @ diff).reshape(
        4,
    )
    l_xx_f = 2 * matrix_Qlambd
    l_x[:] = l_x_f.squeeze()
    l_xx[:, :] = l_xx_f
    return l_x, l_xx


def get_termianl_state_idex(ss_xcurv, Qfun, xt):
    xcurv = ss_xcurv[-1]
    xcurv = np.array(xcurv)
    one_vec = np.ones((xcurv.shape[1],1))
    x0_vec = (np.dot(np.array([xt]).T, one_vec.T)).T
    xcurv = xcurv.T
    diff = xcurv - x0_vec
    norm = la.norm(diff, 1, axis=1)
    min_index = np.argmin(norm)
    return min_index

def solve_qp_problem(matrix_Qfun_selected_tot, xvar_update, ss_point_selected_tot, weight_qp, matrix_Q_qp):
    # Initialize the qp problem
    opti = ca.Opti()
    # define variable of qp problem
    lambd_qp = opti.variable(matrix_Qfun_selected_tot.shape[0])
    # initialize the cost of qp problem
    cost_qp = 0
    # define constraints
    # two constraints: lambda >=0, 1 @ lambda = 1
    opti.subject_to(lambd_qp >= np.zeros(lambd_qp.shape[0]))
    opti.subject_to(mtimes(np.ones((1, lambd_qp.shape[0])), lambd_qp) == 1)
    # calculate the cost of qp problem
    # cost of qp is defined as: 10 * J(x) @ lambda + (D(x)lambda - x_N) Q (D(x)lambda - x_N))
    error = mtimes(np.array(ss_point_selected_tot), lambd_qp) - xvar_update[:, -1]
    cost_qp += mtimes([error.T, matrix_Q_qp, error])
    cost_qp += weight_qp * mtimes(np.array([matrix_Qfun_selected_tot]),  lambd_qp)
    # solve variable lambda
    option = {"print_header":False, "print_time":False, 'min_step_size': 1e-16}
    opti.minimize(cost_qp)
    opti.solver("sqpmethod", option)
    try:
        sol = opti.solve()
        lamb_solution = sol.value(lambd_qp).T
    except RuntimeError:
        lamb_solution = opti.debug.value(lambd_qp).T
        print("solver fail to find the solution, the non-converged solution is used")
    return lamb_solution

def backward_pass(xvar, uvar, xTerminal, dX, lamb, num_horizon, f_x, f_u, matrix_Q, matrix_R, matrix_Qlamb, obstacle):
    # cost derivation
    l_u, l_uu, l_x, l_xx = get_cost_derivation(
        uvar, dX, matrix_Q, matrix_R, num_horizon, xvar, obstacle
    )
    # Value function at last timestep
    matrix_Vx, matrix_Vxx = get_cost_final(
        xvar,
        xTerminal,
        matrix_Qlamb
    )
    # define control modification k and K
    U_DIM = 2
    X_DIM = 4
    matrix_K = np.zeros((U_DIM, X_DIM, num_horizon))
    matrix_k = np.zeros((U_DIM, num_horizon))
    for idx_b in range(num_horizon - 1, -1, -1):
        matrix_Qx = l_x[:, idx_b] + f_x[:,:,idx_b].T @ matrix_Vx
        matrix_Qu = l_u[:, idx_b] + f_u[:,:,idx_b].T @ matrix_Vx
        matrix_Qxx = l_xx[:, :, idx_b] + f_x[:,:,idx_b].T @ matrix_Vxx @ f_x[:,:,idx_b]
        matrix_Quu = l_uu[:, :, idx_b] + f_u[:,:,idx_b].T @ matrix_Vxx @ f_u[:,:,idx_b]
        matrix_Qux = f_u[:,:,idx_b].T @ matrix_Vxx @ f_x[:,:,idx_b]
        # Improved Regularization
        matrix_Quu_evals, matrix_Quu_evecs = np.linalg.eig(matrix_Quu)
        matrix_Quu_evals[matrix_Quu_evals < 0] = 0.0
        matrix_Quu_evals += lamb
        matrix_Quu_inv = np.dot(
            matrix_Quu_evecs, np.dot(np.diag(1.0 / matrix_Quu_evals), matrix_Quu_evecs.T)
        )
        # Calculate feedforward and feedback terms
        matrix_k[:, idx_b] = -matrix_Quu_inv @ matrix_Qu
        matrix_K[:, :, idx_b] = -matrix_Quu_inv @ matrix_Qux
        # Update value function for next time step
        matrix_Vx = matrix_Qx - matrix_K[:, :, idx_b].T @ matrix_Quu @ matrix_k[:, idx_b]
        matrix_Vxx = matrix_Qxx - matrix_K[:, :, idx_b].T @ matrix_Quu @ matrix_K[:, :, idx_b]
    return matrix_k, matrix_K

def forward_pass(xvar, uvar, xTerminal, timestep, num_horizon, matrix_k, matrix_K, matrix_Q, matrix_R, matrix_Qlamb):
    X_DIM = 4
    U_DIM = 2
    xvar_new = np.zeros((X_DIM, num_horizon + 1))
    xvar_new[:, 0] = xvar[:, 0]
    uvar_new = np.zeros((U_DIM, num_horizon))
    cost_new = 0
    for idx_f in range(num_horizon):
        uvar_new[:, idx_f] = (
            uvar[:, idx_f] + matrix_k[:, idx_f] + matrix_K[:, :, idx_f] @ (xvar_new[:, idx_f] - xvar[:, idx_f])
        )
        xvar_new[:, idx_f + 1] = kinetic_bicycle(xvar_new[:,idx_f], uvar_new[:,idx_f], timestep)
        l_state_temp = (xvar_new[:, idx_f] - xTerminal).T @ matrix_Q @ (xvar_new[:, idx_f] - xTerminal)
        l_ctrl_temp = uvar_new[:, idx_f].T @ matrix_R @ uvar_new[:, idx_f]
        cost_new = cost_new + l_state_temp + l_ctrl_temp
    dx_terminal_temp = xvar_new[:, -1] - xTerminal.T
    dx_terminal_temp = dx_terminal_temp.reshape(X_DIM)
    cost_new = cost_new + dx_terminal_temp.T @ matrix_Qlamb @ dx_terminal_temp
    return xvar_new, uvar_new, cost_new