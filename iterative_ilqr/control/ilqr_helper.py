import numpy as np
import scipy.linalg as la
import casadi as ca
from casadi import *
from systems.kinetic_bicycle import *
from utils.constants_kinetic_bicycle import *


def get_cost_derivation(ctrl_U, dX, ilqr_param, num_horizon, xvar, obstacle, sys_param):
    # define control cost derivation
    l_u = np.zeros((U_DIM, num_horizon))
    l_uu = np.zeros((U_DIM, U_DIM, num_horizon))
    l_x = np.zeros((X_DIM, num_horizon))
    l_xx = np.zeros((X_DIM, X_DIM, num_horizon))
    # obstacle avoidance
    if obstacle is not None:
        safety_margin = ilqr_param.safety_margin
        q1 = ilqr_param.tuning_obs_q1
        q2 = ilqr_param.tuning_obs_q2
        # parameters of the obstacle
        xposition = obstacle.x
        yposition = obstacle.y
        a = obstacle.width
        b = obstacle.height
    for i in range(num_horizon):
        b_dot_ctrl, b_ddot_ctrl = add_control_constraint(ctrl_U[:, i], ilqr_param, sys_param)
        l_u_i = (2 * ilqr_param.matrix_R @ ctrl_U[:, i]).reshape(U_DIM, 1) + b_dot_ctrl
        l_uu_i = 2 * ilqr_param.matrix_R + b_ddot_ctrl
        l_x_i = (2 * ilqr_param.matrix_Q @ dX[:, i]).reshape(X_DIM, 1)
        l_xx_i = 2 * ilqr_param.matrix_Q
        # calculate control barrier functions for each obstacle at timestep
        if obstacle is not None:
            degree = 2
            if obstacle.spd == 0 or obstacle.spd is None:
                diffz = xvar[0, i] - xposition
                diffy = xvar[1, i] - yposition
            if obstacle.moving_option == 1:  # moving updward
                diffz = xvar[0, i] - xposition
                diffy = xvar[1, i] - (yposition + i * obstacle.spd)
            matrix_P1 = np.diag([1 / (a ** degree), 1 / (b ** degree), 0, 0])
            if obstacle.moving_option == 2:  # moving towards left
                diffz = xvar[0, i] - (xposition - i * obstacle.spd)
                diffy = xvar[1, i] - (yposition)
            matrix_P1 = np.diag([1 / (a ** degree), 1 / (b ** degree), 0, 0])
            diff = np.array([diffz, diffy, 0, 0]).reshape(-1, 1)
            h = 1 + safety_margin - diff.T @ matrix_P1 @ diff
            h_dot = -2 * matrix_P1 @ diff
            _, b_dot_obs, b_ddot_obs = repelling_cost_function(q1, q2, h, h_dot)
            _, b_dot_obs, b_ddot_obs = repelling_cost_function(q1, q2, h, h_dot)
            l_x_i += b_dot_obs
            l_xx_i += b_ddot_obs
        l_u[:, i] = l_u_i.squeeze()
        l_uu[:, :, i] = l_uu_i.squeeze()
        l_xx[:, :, i] = l_xx_i.squeeze()
        l_x[:, i] = l_x_i.squeeze()
    return l_u, l_uu, l_x, l_xx


def repelling_cost_function(q1, q2, c, c_dot):
    # barrier function shaping
    b = q1 * np.exp(q2 * c)
    b_dot = q1 * q2 * np.exp(q2 * c) * c_dot
    b_ddot = q1 * (q2 ** 2) * np.exp(q2 * c) * np.matmul(c_dot, c_dot.T)
    return b, b_dot, b_ddot


def select_points(ss_xcurv, Qfun, iter, x0, num_ss_points):
    Qfun_vec = Qfun[iter]
    xcurv = ss_xcurv[iter]
    xcurv = np.array(xcurv)
    one_vec = np.ones((xcurv.shape[1], 1))
    x0_vec = (np.dot(np.array([x0]).T, one_vec.T)).T
    xcurv = xcurv.T
    diff = xcurv - x0_vec
    norm = la.norm(diff, 1, axis=1)
    idxMinNorm = np.argsort(norm).astype(int)
    print("idx selected", idxMinNorm[0 : int(num_ss_points)])
    ss_points = xcurv[idxMinNorm[0 : int(num_ss_points)]].T
    sel_Qfun = Qfun_vec[idxMinNorm[0 : int(num_ss_points)]]
    return ss_points, sel_Qfun, idxMinNorm[0 : int(num_ss_points)]


def add_control_constraint(u, ilqr_param, sys_param):
    # Add constraints of the control input:
    matrix_P1 = np.array([[1], [0]])
    matrix_P2 = np.array([[0], [1]])
    q1 = ilqr_param.tuning_ctrl_q1
    q2 = ilqr_param.tuning_ctrl_q2
    # Add acceleration barrier max
    c = np.matmul(u.T, matrix_P1) - ACCEL_MAX
    _, b_dot_1, b_ddot_1 = repelling_cost_function(q1, q2, c, matrix_P1)
    # Add acceleration barrier min
    c = ACCEL_MIN - np.matmul(u.T, matrix_P1)
    _, b_dot_2, b_ddot_2 = repelling_cost_function(q1, q2, c, -matrix_P1)
    # Yawrate Barrier Max
    c = (np.matmul(u.T, matrix_P2)) - DELTA_MAX
    b_3, b_dot_3, b_ddot_3 = repelling_cost_function(q1, q2, c, matrix_P2)
    # Yawrate Barrier Min
    c = DELTA_MIN - np.matmul(u.T, matrix_P2)
    b_4, b_dot_4, b_ddot_4 = repelling_cost_function(q1, q2, c, -matrix_P2)
    b_dot_ctrl = b_dot_1 + b_dot_2 + b_dot_3 + b_dot_4
    b_ddot_ctrl = b_ddot_1 + b_ddot_2 + b_ddot_3 + b_ddot_4
    return b_dot_ctrl, b_ddot_ctrl


def get_cost_final(
    x,
    x_terminal,
    matrix_Qlambd,
    obstacle,
    ilqr_param,
):
    # define variables
    l_x = np.zeros((X_DIM))
    l_xx = np.zeros((X_DIM, X_DIM))
    # Add convex hull constraint as the terminal cost
    # Add state barrier max
    diff = x[:, -1] - x_terminal
    l_x_f = (2 * matrix_Qlambd @ diff).reshape(4, 1)
    l_xx_f = 2 * matrix_Qlambd
    if obstacle is not None:
        safety_margin = ilqr_param.safety_margin
        q1 = ilqr_param.tuning_obs_q1
        q2 = ilqr_param.tuning_obs_q2
        # parameters of the obstacle
        xposition = obstacle.x
        yposition = obstacle.y
        a = obstacle.width
        b = obstacle.height
        degree = 2
        if obstacle.spd == 0 or obstacle.spd is None:
            diffz = x[0, -1] - xposition
            diffy = x[1, -1] - yposition
        if obstacle.moving_option == 1:  # moving updward
            diffz = x[0, -1] - xposition
            diffy = x[1, -1] - (yposition + ilqr_param.num_horizon * obstacle.spd)
        if obstacle.moving_option == 2:  # moving toward left
            diffz = x[0, -1] - (xposition - ilqr_param.num_horizon * obstacle.spd)
            diffy = x[1, -1] - (yposition)
        matrix_P1 = np.diag([1 / (a ** degree), 1 / (b ** degree), 0, 0])
        diff = np.array([diffz, diffy, 0, 0]).reshape(-1, 1)
        h = 1 + safety_margin - diff.T @ matrix_P1 @ diff
        h_dot = -2 * matrix_P1 @ diff
        _, b_dot_obs, b_ddot_obs = repelling_cost_function(q1, q2, h, h_dot)
        _, b_dot_obs, b_ddot_obs = repelling_cost_function(q1, q2, h, h_dot)
        l_x_f += b_dot_obs
        l_xx_f += b_ddot_obs
    l_x[:] = l_x_f.squeeze()
    l_xx[:, :] = l_xx_f.squeeze()
    return l_x, l_xx


def get_termianl_state_idex(ss_xcurv, Qfun, xt):
    xcurv = ss_xcurv[-1]
    xcurv = np.array(xcurv)
    one_vec = np.ones((xcurv.shape[1], 1))
    x0_vec = (np.dot(np.array([xt]).T, one_vec.T)).T
    xcurv = xcurv.T
    diff = xcurv - x0_vec
    norm = la.norm(diff, 1, axis=1)
    min_index = np.argmin(norm)
    return min_index
