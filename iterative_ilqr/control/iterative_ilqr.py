import numpy as np
from copy import deepcopy
from systems.kinetic_bicycle import *
from utils.constants_kinetic_bicycle import *
from control.ilqr_helper import *
import matplotlib.pyplot as plt


def ilqr(
    x,
    u_old,
    ilqr_param,
    num_horizon,
    self_num_horizon,
    xtarget,
    timestep,
    obstacle,
    system_param,
    ss,
    Qfun,
    min_iter,
    iter_total,
    lamb_factor,
    max_lamb,
):
    for iter in range(ilqr_param.max_outloop_iter):
        # Initialize the list which will store the solution to the ftocp for the l-th iteration in the safe set
        cost_list = []
        u_list = []
        id_list = []
        x_pred = []
        u_pred = []
        x_terminal_list = []
        for id in range(min_iter, iter_total):
            # select k neigbors for initial
            lamb = ilqr_param.lamb
            cost_iter = []
            input_iter = []
            x_pred_iter = []
            u_pred_iter = []
            x_terminal_iter = []
            if iter == 0:
                zt = x
            else:
                zt = xvar_optimal[:, -1]
            index_ss_points = select_close_ss(ss, ilqr_param, id, zt)
            # for id_ss in index_ss_points:
            # plt.plot(self.ss[id][0, id_ss], self.ss[id][1, id_ss],'o',color='b',markersize = 15)
            # plt.plot(self.ss[id][0,:],self.ss[id][1,:], 'o',color = 'r')
            for j in index_ss_points:
                # define variables
                uvar = np.zeros((U_DIM, num_horizon))
                xvar = np.zeros((X_DIM, num_horizon + 1))
                xvar[:, 0] = x
                # diffence between xvar and x_track
                dX = np.zeros((X_DIM, num_horizon + 1))
                dX[:, 0] = xvar[:, 0] - xtarget
                x_terminal = ss[id][:, j]
                cost_terminal = Qfun[id][j]
                if self_num_horizon > 1:
                    # Iteration of ilqr for tracking
                    matrix_Q, matrix_Qlamb, matrix_R = (
                        ilqr_param.matrix_Q,
                        ilqr_param.matrix_Qlamb,
                        ilqr_param.matrix_R,
                    )
                    for iter_ilqr in range(ilqr_param.max_ilqr_iter):
                        cost = 0
                        # Forward simulation
                        for idx_f in range(num_horizon):
                            uvar[U_ID["accel"], idx_f] = np.clip(
                                uvar[U_ID["accel"], idx_f], ACCEL_MIN, ACCEL_MAX
                            )
                            uvar[U_ID["delta"], idx_f] = np.clip(
                                uvar[U_ID["delta"], idx_f], DELTA_MIN, DELTA_MAX
                            )
                            xvar[:, idx_f + 1] = kinetic_bicycle(
                                xvar[:, idx_f], uvar[:, idx_f], timestep
                            )
                            dX[:, idx_f + 1] = xvar[:, idx_f + 1] - xtarget.T
                            l_state = (
                                (xvar[:, idx_f] - xtarget).T @ matrix_Q @ (xvar[:, idx_f] - xtarget)
                            )
                            l_ctrl = uvar[:, idx_f].T @ matrix_R @ uvar[:, idx_f]
                            cost = cost + l_state + l_ctrl
                        dx_terminal = xvar[:, -1] - x_terminal.T
                        dx_terminal = dx_terminal.reshape(X_DIM)
                        # ILQR cost is defined as: x_k Q x_k + u_k R u_k + (D(x_t)lambda - x_N) Qlamb (D(x)lambda - x_N)
                        cost = cost + dx_terminal.T @ matrix_Qlamb @ dx_terminal
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
                            self_num_horizon,
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
                    for i in range(1, ilqr_param.max_relax_iter + 1):
                        if np.linalg.norm([xvar[:, -1] - x_terminal]) <= 80.0 * i / (10 ** iter):
                            cost_it = cost_terminal + num_horizon + 100 * i
                            break
                        elif np.linalg.norm(
                            [xvar[:, -1] - x_terminal]
                        ) > 80.0 * ilqr_param.max_relax_iter / (10 ** iter):
                            cost_it = float("Inf")
                            break
                else:
                    x_next = kinetic_bicycle(x, u_old[:, 0], timestep)
                    xvar[:, -1] = x_next
                    uvar[:, 0] = u_old[:, 0]
                    cost_it = 1 + cost_terminal
                    # check for feasibility and store the solution
                    if np.linalg.norm([x_next[:] - x_terminal[:]]) <= ilqr_param.reach_error:
                        cost_it = 1 + cost_terminal
                    else:
                        cost_it = float("Inf")
                # Store the cost and solution associated with xf. From these solution we will pick and apply the best one
                cost_iter.append(cost_it)
                u_pred_iter.append(deepcopy(uvar[:, 0]))
                x_pred_iter.append(deepcopy(xvar))
                input_iter.append(deepcopy(uvar))
                x_terminal_iter.append(deepcopy(x_terminal))
            id_list.append(index_ss_points)
            cost_list.append(cost_iter)
            u_list.append(input_iter)
            x_pred.append(x_pred_iter)
            u_pred.append(u_pred_iter)
            x_terminal_list.append(x_terminal_iter)
        # Pick the best trajectory among the feasible ones
        best_iter_loc_ss = cost_list.index(min(cost_list))
        cost_vec = cost_list[best_iter_loc_ss]
        best_time = cost_vec.index(min(cost_vec))
        best_iter = best_iter_loc_ss + min_iter
        uvar_optimal = u_list[best_iter_loc_ss][best_time]
        xvar_optimal = x_pred[best_iter_loc_ss][best_time]
        xf_optimal = x_terminal_list[best_iter_loc_ss][best_time]
        # for i in range(num_horizon+1):
        #     plt.plot(xvar_optimal[0, i], xvar_optimal[1, i], 's', color = 'black', markersize = 15)
        # plt.plot(xvar_optimal[0,-1], xvar_optimal[1,-1], 'o', color = 'black',markersize = 20)
        # plt.plot(xvar_optimal[0,0], xvar_optimal[1,0], 'o', color = 'black',markersize = 17)
        # plt.plot(xf_optimal[0],xf_optimal[1],'o', color = 'g',markersize = 10)
        # plt.show()
        u_pred = uvar_optimal[:, 0]
        if self_num_horizon > 1:
            u_old_update = uvar_optimal[:, 1:]
        if iter == 2:
            # Change time horizon length
            if (id_list[best_iter_loc_ss][best_time] + 1) > (ss[best_iter].shape[1] - 1):
                self_num_horizon = self_num_horizon - 1
            break
    return u_pred, u_old_update, self_num_horizon


def select_close_ss(ss, ilqr_param, iter, x0):
    x = ss[iter]
    one_vec = np.ones((x.shape[1], 1))
    terminal_guess_vec = np.dot(np.array([x0]).T, one_vec.T)
    # terminal_guess_vec = (np.dot(np.ones((x.shape[1], 1)), np.array([self.x_terminal_guess]))).T
    diff = x - terminal_guess_vec
    norm = la.norm(np.array(diff), 1, axis=0)
    index_min_norm = np.argsort(norm)
    print("k neighbors", index_min_norm[0 : ilqr_param.num_ss_points])
    return index_min_norm[0 : ilqr_param.num_ss_points]


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
        xvar, x_terminal, ilqr_param.matrix_Qlamb, obstacle, ilqr_param
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
    for idx_f in range(num_horizon):
        uvar_new[:, idx_f] = (
            uvar[:, idx_f]
            + matrix_k[:, idx_f]
            + matrix_K[:, :, idx_f] @ (xvar_new[:, idx_f] - xvar[:, idx_f])
        )
        uvar_new[0, idx_f] = np.clip(uvar_new[0, idx_f], ACCEL_MIN, ACCEL_MAX)
        uvar_new[1, idx_f] = np.clip(uvar_new[1, idx_f], DELTA_MIN, DELTA_MAX)
        xvar_new[:, idx_f + 1] = kinetic_bicycle(xvar_new[:, idx_f], uvar_new[:, idx_f], timestep)
        l_state_temp = (
            (xvar_new[:, idx_f] - x_terminal).T
            @ ilqr_param.matrix_Q
            @ (xvar_new[:, idx_f] - x_terminal)
        )
        l_ctrl_temp = uvar_new[:, idx_f].T @ ilqr_param.matrix_R @ uvar_new[:, idx_f]
        cost_new = cost_new + l_state_temp + l_ctrl_temp
    dx_terminal_temp = xvar_new[:, -1] - x_terminal.T
    dx_terminal_temp = dx_terminal_temp.reshape(X_DIM)
    cost_new = cost_new + dx_terminal_temp.T @ ilqr_param.matrix_Qlamb @ dx_terminal_temp
    return xvar_new, uvar_new, cost_new
