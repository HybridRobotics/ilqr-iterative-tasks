import numpy as np
from systems.kinetic_bicycle import *
from utils.constants_kinetic_bicycle import *
from control.ilqr_helper import *


def ilqr(
    ilqr_param,
    num_horizon,
    xtarget,
    timestep,
    obstacle,
    system_param,
    x_terminal,
    dX,
    uvar,
    xvar,
    lamb,
):
    # Iteration of ilqr for tracking
    matrix_Q, matrix_Qterminal, matrix_R = (
        ilqr_param.matrix_Q,
        ilqr_param.matrix_Qterminal,
        ilqr_param.matrix_R,
    )
    # regularization parameters in backwards
    lamb_factor = ilqr_param.lamb_factor
    max_lamb = ilqr_param.max_lamb
    for iter_ilqr in range(ilqr_param.max_ilqr_iter):
        cost = 0
        # Forward simulation
        for idx in range(num_horizon):
            uvar[U_ID["accel"], idx] = np.clip(
                uvar[U_ID["accel"], idx], -system_param.a_max, system_param.a_max
            )
            uvar[U_ID["delta"], idx] = np.clip(
                uvar[U_ID["delta"], idx],
                -round(system_param.delta_max, 2),
                round(system_param.delta_max, 2),
            )
            xvar[:, idx + 1] = kinetic_bicycle(xvar[:, idx], uvar[:, idx], timestep)
            dX[:, idx + 1] = xvar[:, idx + 1] - xtarget.T
            l_state = (xvar[:, idx] - xtarget).T @ matrix_Q @ (xvar[:, idx] - xtarget)
            l_ctrl = uvar[:, idx].T @ matrix_R @ uvar[:, idx]
            cost = cost + l_state + l_ctrl
        dx_terminal = (xvar[:, -1] - x_terminal.T).reshape(X_DIM)
        # ILQR cost is defined as: x_k Q x_k + u_k R u_k + (D(x_t)lambda - x_N) Qlamb (D(x)lambda - x_N)
        cost = cost + dx_terminal.T @ matrix_Qterminal @ dx_terminal
        # Backward pass
        matrix_k, matrix_K = backward_pass(
            xvar,
            uvar,
            x_terminal,
            dX,
            lamb,
            num_horizon,
            timestep,
            ilqr_param,
            obstacle,
            system_param,
        )
        # Forward pass
        xvar_new, uvar_new, cost_new = forward_pass(
            xvar,
            uvar,
            x_terminal,
            ilqr_param,
            timestep,
            num_horizon,
            matrix_k,
            matrix_K,
            system_param,
        )
        if cost_new < cost:
            uvar = uvar_new
            xvar = xvar_new
            lamb /= lamb_factor
            if abs((cost_new - cost) / cost) < ilqr_param.eps:
                print("Convergence achieved")
                break
        else:
            lamb *= lamb_factor
            if lamb > max_lamb:
                break
    return uvar, xvar, lamb


def backward_pass(
    xvar, uvar, x_terminal, dX, lamb, num_horizon, timestep, ilqr_param, obstacle, sys_param
):
    # System derivation
    f_x = get_A_matrix(
        xvar[X_ID["v"], 1:],
        xvar[X_ID["theta"], 1:],
        uvar[U_ID["accel"], :],
        num_horizon,
        timestep,
    )
    f_u = get_B_matrix(xvar[X_ID["theta"], 1:], num_horizon, timestep)
    # cost derivation
    l_u, l_uu, l_x, l_xx = get_cost_derivation(
        uvar, dX, ilqr_param, num_horizon, xvar, obstacle, sys_param
    )
    # Value function at last timestep
    matrix_Vx, matrix_Vxx = get_cost_final(
        xvar, x_terminal, ilqr_param.matrix_Qterminal, obstacle, ilqr_param
    )
    # define control modification k and K
    matrix_K = np.zeros((U_DIM, X_DIM, num_horizon))
    matrix_k = np.zeros((U_DIM, num_horizon))
    for idx_b in range(num_horizon - 1, -1, -1):
        matrix_Qx = l_x[:, idx_b] + f_x[:, :, idx_b].T @ matrix_Vx
        matrix_Qu = l_u[:, idx_b] + f_u[:, :, idx_b].T @ matrix_Vx
        matrix_Qxx = l_xx[:, :, idx_b] + f_x[:, :, idx_b].T @ matrix_Vxx @ f_x[:, :, idx_b]
        matrix_Quu = l_uu[:, :, idx_b] + f_u[:, :, idx_b].T @ matrix_Vxx @ f_u[:, :, idx_b]
        matrix_Qux = f_u[:, :, idx_b].T @ matrix_Vxx @ f_x[:, :, idx_b]
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


def forward_pass(
    xvar, uvar, x_terminal, ilqr_param, timestep, num_horizon, matrix_k, matrix_K, sys_param
):
    xvar_new = np.zeros((X_DIM, num_horizon + 1))
    xvar_new[:, 0] = xvar[:, 0]
    uvar_new = np.zeros((U_DIM, num_horizon))
    cost_new = 0
    for idx in range(num_horizon):
        uvar_new[:, idx] = (
            uvar[:, idx]
            + matrix_k[:, idx]
            + matrix_K[:, :, idx] @ (xvar_new[:, idx] - xvar[:, idx])
        )
        uvar_new[0, idx] = np.clip(uvar_new[0, idx], -sys_param.a_max, sys_param.a_max)
        uvar_new[1, idx] = np.clip(
            uvar_new[1, idx], -round(sys_param.delta_max, 2), round(sys_param.delta_max, 2)
        )
        xvar_new[:, idx + 1] = kinetic_bicycle(xvar_new[:, idx], uvar_new[:, idx], timestep)
        l_state = (
            (xvar_new[:, idx] - x_terminal).T
            @ ilqr_param.matrix_Q
            @ (xvar_new[:, idx] - x_terminal)
        )
        l_ctrl = uvar_new[:, idx].T @ ilqr_param.matrix_R @ uvar_new[:, idx]
        cost_new = cost_new + l_state + l_ctrl
    dx_terminal = (xvar_new[:, -1] - x_terminal.T).reshape(X_DIM)
    cost_new = cost_new + dx_terminal.T @ ilqr_param.matrix_Qterminal @ dx_terminal
    return xvar_new, uvar_new, cost_new
