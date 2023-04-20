import numpy as np
import datetime
from copy import deepcopy
from utils.constants_kinetic_bicycle import *
from systems.kinetic_bicycle import *
from numpy import linalg as la
from control.nonlinear_lmpc import *
from control.ilqr_helper import *
from control.iterative_ilqr import *
import matplotlib.pyplot as plt
import control.ilqr_helper
import pickle as pkl


class KineticBicycleParam:
    def __init__(self, delta_max=np.pi / 2, a_max=2.0, v_max=10, v_min=0):
        self.delta_max = delta_max
        self.a_max = a_max
        self.v_max = v_max
        self.v_min = v_min


class Obstacle:
    def __init__(self, x, y, width, height, spd=None, timestep=None, moving_option=None):
        self.x0 = self.x = x
        self.y0 = self.y = y
        self.width = width
        self.height = height
        self.spd = spd
        self.timestep = timestep
        self.data = {}
        self.data["state"] = []
        self.states = np.array([self.x0, self.y0])
        self.moving_option = moving_option  # 1: moving upwards # 2: moving towards left

    def plot_obstacle(self):
        count = 0
        for data in self.data["state"][-1]:
            x_obs = []
            y_obs = []
            for index in np.linspace(0, 2 * np.pi, 1000):
                x_obs.append(data[0] + self.width * np.cos(index))
                y_obs.append(data[1] + self.height * np.sin(index))
            if count % 5 == 0:
                plt.plot(x_obs, y_obs, "-k", label="Obstacle", linewidth=5, alpha=1 - count / 90)
            count += 1

    def update_obstacle(self):
        if self.spd is not None:
            if self.spd == 0:
                pass
            if self.moving_option == 1:
                self.y += self.spd * self.timestep
            if self.moving_option == 2:
                self.x -= self.spd * self.timestep
        self.states = np.vstack((self.states, [self.x, self.y]))

    def reset_obstacle(self):
        self.x = self.x0
        self.y = self.y0
        self.data["state"].append(self.states)
        self.states = np.array([self.x0, self.y0])


class KineticBicycle:
    def __init__(self, direct_ctrl_policy=False, system_param=None):
        self.system_param = system_param
        self.direct_ctrl_policy = direct_ctrl_policy
        self.time = 0.0
        self.delta_timer = None
        self.feasible = None
        self.timestep = None
        self.x = None
        self.u = None
        self.zero_noise_flag = False
        self.states, self.inputs, self.timestamps = None, None, None
        self.solver_times, self.feasibility = None, None
        data_names = {"state", "input", "timestamp"}
        diagnostics_names = {"solver_time", "feasibility"}
        self.data, self.diagnostics = {}, {}
        for data_name in data_names:
            self.data[data_name] = []
        for diagnostic_name in diagnostics_names:
            self.diagnostics[diagnostic_name] = []
        self.iters = 0

    def set_zero_noise(self):
        self.zero_noise_flag = True

    def set_timestep(self, dt):
        self.timestep = dt

    def set_state(self, x):
        self.x = x
        self.states = x
        self.timestamps = None
        self.inputs = None
        self.solver_times = None
        self.feasible = None

    def get_traj(self):
        if self.direct_ctrl_policy == False:
            angle = np.pi / 6
            total_time_steps = int(120 / self.timestep)
            xcl = np.zeros((X_DIM,))
            ucl = None
            u = np.zeros((U_DIM,))
            for i in range(0, total_time_steps):
                if i <= 1 / self.timestep:
                    u[U_ID["accel"]] = 1
                elif i >= (total_time_steps - 4 / self.timestep) and i <= (
                    total_time_steps - 3 / self.timestep
                ):
                    u[U_ID["accel"]] = -1
                else:
                    u[U_ID["accel"]] = 0

                if i > 0 and i <= 1 / self.timestep:
                    u[U_ID["delta"]] = angle
                elif (
                    i >= total_time_steps / 2 - 2 / self.timestep
                    and i <= total_time_steps / 2 - 1 / self.timestep
                ):
                    u[U_ID["delta"]] = -angle
                else:
                    u[U_ID["delta"]] = 0
                xcl = np.vstack(
                    (xcl, kinetic_bicycle(xcl if i == 0 else xcl[-1, :], u, self.timestep))
                )
                ucl = u if ucl is None else np.vstack((ucl, u))
            np.savetxt("data/closed_loop_feasible.txt", xcl, fmt="%f")
        else:
            xcl = np.loadtxt("data/closed_loop_multi_laps.txt")
            ucl = np.loadtxt("data/input_multi_laps.txt")
        self.xcl = xcl
        self.ucl = ucl

    def set_ctrl_policy(self, ctrl_policy):
        self.ctrl_policy = ctrl_policy

    def calc_ctrl_input(self):
        self.ctrl_policy.set_state(self.x)
        startTimer = datetime.datetime.now()
        try:
            self.ctrl_policy.calc_input()
            self.u = self.ctrl_policy.get_input()
            self.delta_timer = (datetime.datetime.now() - startTimer).total_seconds()
            print("time to solve:{}".format(self.delta_timer))
            self.feasible = 1

        except RuntimeError:
            self.feasible = 0
            print("solver fail to find the solution")

    def forward_one_step(self):
        self.calc_ctrl_input()
        self.forward_dynamics()
        self.ctrl_policy.set_state(self.x)
        self.update_memory()

    def update_memory(self):
        self.states = np.vstack((self.states, self.x))
        self.inputs = self.u if self.inputs is None else np.vstack((self.inputs, self.u))
        self.timestamps = (
            self.time if self.timestamps is None else np.vstack((self.timestamps, self.time))
        )

        if (
            self.timestamps is not None
            and type(self.timestamps) is not float
            and len(self.timestamps) >= 121
        ):
            with open("data/ego_nlmpc_ss_" + str(8) + "_add_static_obstacle.obj", "wb") as handle:
                pkl.dump(self, handle, protocol=pkl.HIGHEST_PROTOCOL)

        self.solver_times = (
            self.delta_timer
            if self.solver_times is None
            else np.vstack((self.solver_times, self.delta_timer))
        )
        self.feasibility = (
            self.feasible
            if self.feasibility is None
            else np.vstack((self.feasibility, self.feasible))
        )

    def update_memory_post_iter(self):
        self.data["state"].append(self.states)
        self.data["input"].append(self.inputs)
        self.data["timestamp"].append(self.timestamps)
        self.diagnostics["solver_time"].append(self.solver_times)
        self.diagnostics["feasibility"].append(self.feasibility)
        self.set_state(np.zeros((X_DIM,)))

    def forward_dynamics(self):
        # This function computes the system evolution. Note that the discretization is delta_t and therefore is needed that
        # dt <= delta_t and ( dt / delta_t) = integer value
        # Discretization Parameters
        delta_t = 0.001
        x_next = np.zeros((X_DIM))
        x_next = self.x
        # Initialize counter
        x_next = kinetic_bicycle(x_next, self.u, self.timestep)
        # Noises
        if not self.zero_noise_flag:
            noise_v = np.maximum(-0.05, np.minimum(np.random.randn() * 0.01, 0.05))
            noise_theta = np.maximum(-0.05, np.minimum(np.random.randn() * 0.005, 0.05))
            x_next[X_ID["v"]] = x_next[X_ID["v"]] + 0.5 * noise_v
            x_next[X_ID["theta"]] = x_next[X_ID["theta"]] + 0.5 * noise_theta
        self.x = x_next
        self.time += self.timestep


class ControlBase:
    def __init__(self):
        self.time = 0.0
        self.timestep = None
        self.x = None
        self.u = None
        self.iters = 0

    def set_timestep(self, timestep):
        self.timestep = timestep

    def set_state(self, x):
        self.x = x

    def calc_input(self):
        pass

    def get_input(self):
        return self.u


# obs_q1, obs_q2 2.05 for static obstacle, add static obstacle
# obs_q1, obs_q2 2.74 for add moving obstacle with option 1
# safety_margin = 0


class iLqrParam:
    def __init__(
        self,
        matrix_Q=0 * np.diag([0.0, 0.0, 0.0, 0.0]),
        matrix_R=0 * np.diag([0.05, 0.05]),
        matrix_Qterminal=2 * np.diag([1.0, 1.0, 20.0, 0.02]),
        num_ss_points=8,
        num_ss_iter=1,
        num_horizon=6,
        tuning_state_q1=1.0,
        tuning_state_q2=1.0,
        tuning_ctrl_q1=1.0,
        tuning_ctrl_q2=1.0,
        tuning_obs_q1=2.74,
        tuning_obs_q2=2.74,
        safety_margin=0.0,
        max_ilqr_iter=150,
        eps=1e-2,
        lamb=1,
        lamb_factor=10,
        max_lamb=1000,
        reach_error=1.0,
        max_relax_iter=55,
        max_outloop_iter=50,
        timestep=None,
        lap_number=None,
        time_ilqr=None,
        ss_option=None,
        all_ss_point=False,
        all_ss_iter=False,
    ):
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R
        self.matrix_Qterminal = matrix_Qterminal
        self.num_ss_points = num_ss_points
        self.num_ss_iter = num_ss_iter
        self.num_horizon = num_horizon
        self.timestep = timestep
        self.lap_number = lap_number
        self.time_ilqr = time_ilqr
        self.ss_option = ss_option
        self.all_ss_point = all_ss_point
        self.all_ss_iter = all_ss_iter
        self.tuning_state_q1 = tuning_state_q1
        self.tuning_state_q2 = tuning_state_q2
        self.tuning_ctrl_q1 = tuning_ctrl_q1
        self.tuning_ctrl_q2 = tuning_ctrl_q2
        self.tuning_obs_q1 = tuning_obs_q1
        self.tuning_obs_q2 = tuning_obs_q2
        self.safety_margin = safety_margin
        self.max_ilqr_iter = max_ilqr_iter
        # rate of convergence for ILQR
        self.eps = eps
        # regularization factor for ILQR
        self.lamb = lamb
        self.lamb_factor = lamb_factor
        self.max_lamb = max_lamb
        self.reach_error = reach_error
        # maximum iteration of weights' tunning
        self.max_relax_iter = max_relax_iter
        self.max_outloop_iter = max_outloop_iter


class iLqr(ControlBase):
    def __init__(self, ilqr_param, obstacle=None, system_param=None):
        ControlBase.__init__(self)
        self.ilqr_param = ilqr_param
        self.system_param = system_param
        self.ss = []
        self.u_ss = []
        self.Qfun = []
        self.ss_point_selected_id = []
        self.x_terminal_guess = None
        self.x_guess = None
        self.iter = 0
        self.iter_cost = []
        self.itCost = []
        self.cost = None
        self.old_cost = None
        self.old_iter = None
        self.u_old = None
        self.x_pred = None
        self.u_pred = None
        self.cost_improve = None
        self.num_horizon = self.ilqr_param.num_horizon
        self.matrix_Qterminal = self.ilqr_param.matrix_Qterminal
        self.matrix_Q = self.ilqr_param.matrix_Q
        self.matrix_R = self.ilqr_param.matrix_R
        self.obstacle = obstacle

    def select_close_ss(self, iter, x0):
        x = self.ss[iter]
        one_vec = np.ones((x.shape[1], 1))
        terminal_guess_vec = np.dot(np.array([x0]).T, one_vec.T)
        # terminal_guess_vec = (np.dot(np.ones((x.shape[1], 1)), np.array([self.x_terminal_guess]))).T
        diff = x - terminal_guess_vec
        norm = la.norm(np.array(diff), 1, axis=0)
        index_min_norm = np.argsort(norm)
        print("k neighbors", index_min_norm[0 : self.ilqr_param.num_ss_points])
        return index_min_norm[0 : self.ilqr_param.num_ss_points]

    def add_trajectory(self, x, u):
        self.ss.append(deepcopy(x.T))
        self.u_ss.append(deepcopy(u.T))
        self.Qfun.append(deepcopy(np.arange(x.shape[0] - 1, -1, -1)))
        self.num_horizon = self.ilqr_param.num_horizon
        self.x_terminal_guess = x.T[:, self.num_horizon]
        self.cost = self.Qfun[-1][0]
        self.iter_cost.append(deepcopy(self.cost))
        self.old_cost = self.cost + 1
        self.old_iter = self.iter
        self.x_sol = x.T[:, 0 : (self.num_horizon + 1)]
        self.u_sol = u.T[:, 0 : (self.num_horizon)]
        self.cost_improve = -1
        self.iter = self.iter + 1
        self.ss_point_selected_id = []
        min_cost = np.min(self.iter_cost)
        for id in range(self.iter):
            iter_cost = np.shape(self.ss[id])[1] - 1
            if self.ilqr_param.all_ss_point:
                self.ss_point_selected_id.append(np.arange(0, self.ss[id].shape[1]))
            else:
                self.ss_point_selected_id.append(
                    np.arange(
                        iter_cost - min_cost + self.num_horizon,
                        iter_cost - min_cost + self.num_horizon + self.ilqr_param.num_ss_points,
                    )
                )

    def calc_input(self):
        num_horizon = self.num_horizon
        # tracking when matrix_Q is not zero, this is not used right now
        xtarget = np.array([0, 0, 0, 0])
        # select the oldest iteration used
        min_iter = np.max([0, self.iter - self.ilqr_param.num_ss_iter])
        if self.num_horizon < self.ilqr_param.num_horizon:
            self.u_pred = self.u_old
            self.u = self.u_pred[:, 0]
            self.u_old = self.u_pred[:, 1:]
            self.num_horizon = self.num_horizon - 1
            print("state", self.x)
        else:
            for iter in range(self.ilqr_param.max_outloop_iter):
                # Initialize the list which will store the solution to the ftocp for the l-th iteration in the safe set
                cost_list = []
                u_list = []
                id_list = []
                x_pred = []
                u_pred = []
                for id in range(min_iter, self.iter):
                    # select k neigbors for initial
                    lamb = self.ilqr_param.lamb
                    cost_iter = []
                    input_iter = []
                    x_pred_iter = []
                    u_pred_iter = []
                    if iter == 0:
                        self.x_guess = self.x
                    else:
                        self.x_guess = self.x_pred[:, -1]
                    index_ss_points = self.select_close_ss(id, self.x_guess)
                    for j in index_ss_points:
                        # define variables
                        uvar = np.zeros((U_DIM, num_horizon))
                        xvar = np.zeros((X_DIM, num_horizon + 1))
                        xvar[:, 0] = self.x
                        # diffence between xvar and x_track
                        dX = np.zeros((X_DIM, num_horizon + 1))
                        dX[:, 0] = xvar[:, 0] - xtarget
                        x_terminal = self.ss[id][:, j]
                        cost_terminal = self.Qfun[id][j]
                        if self.num_horizon > 1:
                            uvar, xvar, lamb = ilqr(
                                self.ilqr_param,
                                self.num_horizon,
                                xtarget,
                                self.timestep,
                                self.obstacle,
                                self.system_param,
                                x_terminal,
                                dX,
                                uvar,
                                xvar,
                                lamb,
                            )
                            for i in range(1, self.ilqr_param.max_relax_iter + 1):
                                if np.linalg.norm([xvar[:, -1] - x_terminal]) <= 80.0 * i / (
                                    10 ** iter
                                ):
                                    cost_it = cost_terminal + num_horizon + 100 * i
                                    break
                                elif np.linalg.norm(
                                    [xvar[:, -1] - x_terminal]
                                ) > 80.0 * self.ilqr_param.max_relax_iter / (10 ** iter):
                                    cost_it = float("Inf")
                                    break
                        else:
                            x_next = kinetic_bicycle(self.x, self.u_old[:, 0], self.timestep)
                            xvar[:, -1] = x_next
                            uvar[:, 0] = self.u_old[:, 0]
                            cost_it = 1 + cost_terminal
                            # check for feasibility and store the solution
                            if (
                                np.linalg.norm([x_next[:] - x_terminal[:]])
                                <= self.ilqr_param.reach_error
                            ):
                                cost_it = 1 + cost_terminal
                            else:
                                cost_it = float("Inf")
                        # Store the cost and solution associated with xf. From these solution we will pick and apply the best one
                        cost_iter.append(cost_it)
                        u_pred_iter.append(deepcopy(uvar[:, 0]))
                        x_pred_iter.append(deepcopy(xvar))
                        input_iter.append(deepcopy(uvar))
                    id_list.append(index_ss_points)
                    cost_list.append(cost_iter)
                    u_pred.append(input_iter)
                    x_pred.append(x_pred_iter)
                    u_list.append(u_pred_iter)
                # Pick the best trajectory among the feasible ones
                best_iter_loc_ss = cost_list.index(min(cost_list))
                cost_vec = cost_list[best_iter_loc_ss]
                best_time = cost_vec.index(min(cost_vec))
                best_iter = best_iter_loc_ss + min_iter
                self.u_pred = u_pred[best_iter_loc_ss][best_time]
                self.x_pred = x_pred[best_iter_loc_ss][best_time]
                self.u = self.u_pred[:, 0]
                self.x_terminal_guess = self.x_pred[:, -1]
                if self.num_horizon > 1:
                    self.u_old = self.u_pred[:, 1:]
                if iter == 2:
                    # Change time horizon length
                    if (id_list[best_iter_loc_ss][best_time] + 1) > (
                        self.ss[best_iter].shape[1] - 1
                    ):
                        self.num_horizon = self.num_horizon - 1
                    break
        self.time += self.timestep


class LMPCParam:
    def __init__(
        self,
        matrix_Q=0 * np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        matrix_R=1 * np.diag([1.0, 0.25]),
        matrix_Qslack=5 * np.diag([10, 0, 0, 1, 10, 0]),
        matrix_dR=5 * np.diag([0.8, 0.0]),
        num_ss_points=8,
        num_ss_iter=1,
        num_horizon=6,
        timestep=None,
        lap_number=None,
        time_lmpc=None,
        ss_option=None,
        all_ss_point=False,
        all_ss_iter=False,
    ):
        self.matrix_Q = matrix_Q
        self.matrix_R = matrix_R
        self.matrix_Qslack = matrix_Qslack
        self.matrix_dR = matrix_dR
        self.num_ss_points = num_ss_points
        self.num_ss_iter = num_ss_iter
        self.num_horizon = num_horizon
        self.timestep = timestep
        self.lap_number = lap_number
        self.time_lmpc = time_lmpc
        self.ss_option = ss_option
        self.all_ss_point = all_ss_point
        self.all_ss_iter = all_ss_iter


class LMPC(ControlBase):
    def __init__(self, lmpc_param, obstacle=None, system_param=None, ego=None):
        ControlBase.__init__(self)
        self.lmpc_param = lmpc_param
        self.system_param = system_param
        self.ss = []
        self.u_ss = []
        self.Qfun = []
        self.ss_point_selected_id = []
        self.x_terminal_guess = None
        self.x_guess = None
        self.iter = 0
        self.iter_cost = []
        self.cost = None
        self.old_cost = None
        self.old_iter = None
        self.x_pred = None
        self.u_pred = None
        self.cost_improve = None
        self.num_horizon = self.lmpc_param.num_horizon
        self.obstacle = obstacle
        self.ego = ego

    def select_time_varying_ss(self, iter):
        selected_id = self.ss_point_selected_id[iter]
        selected_id_valid = selected_id[
            (selected_id > 0) & (selected_id < np.shape(self.ss[iter])[1])
        ]
        self.ss_point_selected_id[iter] = self.ss_point_selected_id[iter] + 1
        if np.shape(selected_id_valid)[0] < 1:
            selected_id_valid = np.array([np.shape(self.ss[iter])[1] - 1])
        print("k neighbors", selected_id_valid)
        return selected_id_valid

    def select_close_ss(self, iter):
        x = self.ss[iter]
        terminal_guess_vec = (np.dot(np.ones((x.shape[1], 1)), np.array([self.x_terminal_guess]))).T
        diff = x - terminal_guess_vec
        norm = la.norm(np.array(diff), 1, axis=0)
        index_min_norm = np.argsort(norm)
        print("k neighbors", index_min_norm[0 : self.lmpc_param.num_ss_points])
        return index_min_norm[0 : self.lmpc_param.num_ss_points]

    def calc_input(self, verbose=5):
        u_list = []
        cost_list = []
        id_list = []
        x_pred = []
        u_pred = []
        print("state at current step", self.x)
        if self.lmpc_param.all_ss_iter:
            min_iter = 0
        else:
            min_iter = np.max([0, self.iter - self.lmpc_param.num_ss_iter])
        for id in range(min_iter, self.iter):
            if self.lmpc_param.all_ss_point:
                index_ss_points = np.arange(0, self.ss[id].shape[1])
                print("k neighbors", index_ss_points)
            elif self.lmpc_param.ss_option == "timeVarying":
                index_ss_points = self.select_time_varying_ss(id)
            elif self.lmpc_param.ss_option == "spaceVarying":
                index_ss_points = self.select_close_ss(id)
            cost_iter = []
            input_iter = []
            x_pred_iter = []
            u_pred_iter = []
            for id_point in index_ss_points:
                x_terminal = self.ss[id][:, id_point]
                cost_terminal = self.Qfun[id][id_point]
                self.x_sol, self.u_sol, cost = nlmpc(
                    self.x.tolist(),
                    self.x_guess,
                    x_terminal,
                    self.x_sol,
                    self.u_sol,
                    self.timestep,
                    self.num_horizon,
                    self.old_cost,
                    cost_terminal,
                    self.system_param,
                    self.obstacle,
                )
                cost_iter.append(cost)
                input_iter.append(self.u_sol[:, 0])
                x_pred_iter.append(self.x_sol)
                u_pred_iter.append(self.u_sol)
            id_list.append(index_ss_points)
            cost_list.append(cost_iter)
            u_list.append(input_iter)
            x_pred.append(x_pred_iter)
            u_pred.append(u_pred_iter)
        best_iter_loc_ss = cost_list.index(min(cost_list))
        cost_vec = cost_list[best_iter_loc_ss]
        if min(cost_vec) == float("inf"):
            print("cost vec", cost_vec)
            os.system("pause")
        best_time = cost_vec.index(min(cost_vec))
        best_iter = best_iter_loc_ss + min_iter
        self.u = u_list[best_iter_loc_ss][best_time]
        self.x_pred = x_pred[best_iter_loc_ss][best_time]
        self.u_pred = u_pred[best_iter_loc_ss][best_time]
        self.cost = cost_list[best_iter_loc_ss][best_time]
        self.old_iter = best_iter_loc_ss
        self.cost_improve = self.cost_improve + self.old_cost - self.cost - 1
        self.old_cost = self.cost
        x_pred_flatten = self.x_pred[:, 0 : (self.num_horizon + 1)].T.flatten()
        u_pred_flatten = self.u_pred[:, 0 : (self.num_horizon)].T.flatten()
        if (id_list[best_iter_loc_ss][best_time] + 1) <= (self.ss[best_iter].shape[1] - 1):
            self.x_terminal_guess = self.ss[best_iter][:, id_list[best_iter_loc_ss][best_time] + 1]
            self.x_guess[0 : X_DIM * self.num_horizon] = x_pred_flatten[
                X_DIM : X_DIM * (self.num_horizon + 1)
            ]
            self.x_guess[
                X_DIM * self.num_horizon : X_DIM * (self.num_horizon + 1)
            ] = self.x_terminal_guess
            self.x_guess[
                X_DIM
                * (self.num_horizon + 1) : (
                    X_DIM * (self.num_horizon + 1) + U_DIM * (self.num_horizon - 1)
                )
            ] = u_pred_flatten[U_DIM : U_DIM * self.num_horizon]
            self.x_guess[
                (X_DIM * (self.num_horizon + 1) + U_DIM * (self.num_horizon - 1)) : (
                    X_DIM * (self.num_horizon + 1) + U_DIM * (self.num_horizon)
                )
            ] = self.u_ss[best_iter][:, id_list[best_iter_loc_ss][best_time]]
        else:
            self.x_terminal_guess = x_pred_flatten[
                X_DIM * self.num_horizon : X_DIM * (self.num_horizon + 1)
            ]
            self.x_guess = np.zeros((self.num_horizon) * X_DIM + (self.num_horizon - 1) * U_DIM)
            self.x_guess[0 : X_DIM * self.num_horizon] = x_pred_flatten[
                X_DIM : X_DIM * (self.num_horizon + 1)
            ]
            self.x_guess[
                X_DIM
                * self.num_horizon : (X_DIM * self.num_horizon + U_DIM * (self.num_horizon - 1))
            ] = u_pred_flatten[U_DIM : U_DIM * self.num_horizon]
            if verbose > 0:
                print("Changing horizon to ", self.num_horizon - 1)
            self.num_horizon = self.num_horizon - 1
        self.time += self.timestep

    def add_trajectory(self, x, u):
        self.ss.append(x.T)
        self.u_ss.append(u.T)
        self.Qfun.append(np.arange(x.shape[0] - 1, -1, -1))
        self.num_horizon = self.lmpc_param.num_horizon
        self.x_terminal_guess = x.T[:, self.num_horizon]
        self.x_guess = np.concatenate(
            (
                x[0 : (self.num_horizon + 1), :].flatten(),
                u[0 : (self.num_horizon), :].flatten(),
            ),
            axis=0,
        )
        self.cost = self.Qfun[-1][0]
        self.iter_cost.append(self.cost)
        self.old_cost = self.cost + 1
        self.old_iter = self.iter
        self.x_sol = x.T[:, 0 : (self.num_horizon + 1)]
        self.u_sol = u.T[:, 0 : (self.num_horizon)]
        self.cost_improve = -1
        self.iter = self.iter + 1
        self.ss_point_selected_id = []
        min_cost = np.min(self.iter_cost)
        for id in range(self.iter):
            iter_cost = np.shape(self.ss[id])[1] - 1
            if self.lmpc_param.all_ss_point:
                self.ss_point_selected_id.append(np.arange(0, self.ss[id].shape[1]))
            else:
                self.ss_point_selected_id.append(
                    np.arange(
                        iter_cost - min_cost + self.num_horizon,
                        iter_cost - min_cost + self.num_horizon + self.lmpc_param.num_ss_points,
                    )
                )


class Simulator:
    def __init__(self):
        self.initial_traj = None
        self.robotic = None
        self.timestep = None

    def set_timestep(self, dt):
        self.timestep = dt

    def set_robotic(self, robotic):
        self.robotic = robotic

    def set_traj(self):
        self.initial_traj = self.robotic.xcl

    def sim(self, iter, sim_time=121.0):
        sim_time = 121
        for i in range(0, int(sim_time / self.timestep)):
            # update system state
            self.robotic.forward_one_step()
            if self.robotic.ctrl_policy.obstacle is not None:
                self.robotic.ctrl_policy.obstacle.update_obstacle()
            if np.linalg.norm(self.robotic.x - self.initial_traj[-1, :]) <= 0.8:  # 1e-4
                print("state", self.robotic.x)
                self.robotic.update_memory_post_iter()
                if self.robotic.ctrl_policy.obstacle is not None:
                    self.robotic.ctrl_policy.obstacle.reset_obstacle()
                print("iteration: {}".format(iter) + " finished")
                break
            if i == int(sim_time / self.timestep) - 1:
                self.robotic.update_memory_post_iter()
                if self.robotic.ctrl_policy.obstacle is not None:
                    self.robotic.ctrl_policy.obstacle.reset_obstacle()
                print("iteration: {}".format(iter) + " not finished")

    def plot_inputs(self):
        fig, axs = plt.subplots(2, figsize=(8, 7))
        list_inputs = None
        for iter in range(len(self.robotic.data["input"])):
            list_inputs = (
                self.robotic.data["input"][0]
                if list_inputs is None
                else np.vstack((list_inputs, self.robotic.data["input"][iter]))
            )
        list_times = np.arange(0, len(list_inputs))
        axs[0].plot(list_times, list_inputs[:, U_ID["accel"]], "-o", linewidth=1, markersize=1)
        axs[0].set_xlabel("time [s]", fontsize=14)
        axs[0].set_ylabel("$a$ [m/s^2]", fontsize=14)
        axs[1].plot(list_times, list_inputs[:, U_ID["delta"]], "-o", linewidth=1, markersize=1)
        axs[1].set_xlabel("time [s]", fontsize=14)
        axs[1].set_ylabel("$/delta$ [rad]", fontsize=14)
        plt.show()

    def plot_simulation(self):
        list_states = self.robotic.data["state"][-1]
        fig, ax = plt.subplots()
        if self.robotic.ctrl_policy.obstacle is not None:
            self.robotic.ctrl_policy.obstacle.plot_obstacle()
        (line1,) = ax.plot(
            list_states[:, X_ID["x"]],
            list_states[:, X_ID["y"]],
            label="trajectory at last iteration",
        )
        (line2,) = ax.plot(
            self.robotic.xcl[:, X_ID["x"]],
            self.robotic.xcl[:, X_ID["y"]],
            label="initial trajectory",
        )
        plt.legend(handles=[line1, line2])
        plt.show()
