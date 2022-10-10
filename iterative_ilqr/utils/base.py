import numpy as np
from copy import deepcopy
from utils.constants_kinetic_bicycle import *
from systems.kinetic_bicycle import *
from numpy import linalg as la
import pdb
from control.nonlinear_lmpc import *
import matplotlib.pyplot as plt

class KineticBicycleParam:
    def __init__(self, delta_max=np.pi / 2, a_max=2.0, v_max=10, v_min=0):
        self.delta_max = delta_max
        self.a_max = a_max
        self.v_max = v_max
        self.v_min = v_min
        
class Obstacle:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height        
    
    def plot_obstacle(self):
        x_obs = []
        y_obs = []
        for index in np.linspace(0,2*np.pi,1000):
            x_obs.append(self.x + self.width*np.cos(index))
            y_obs.append(self.y + self.height*np.sin(index))
        plt.plot(x_obs, y_obs, '-k', label="Obstacle")
        
        

class KineticBicycle:
    def __init__(self, system_param=None):
        self.system_param = system_param
        self.time = 0.0
        self.timestep = None
        self.x = None
        self.u = None
        self.zero_noise_flag = False
        self.all_times, self.all_xs, self.all_inputs = [], [], []
        self.times, self.xs, self.inputs = [], [], []
        self.xs.append(np.zeros(X_DIM))
        self.iters = 0
        self.solver_times, self.solver_times = [], []

    def set_zero_noise(self):
        self.zero_noise_flag = True

    def set_timestep(self, dt):
        self.timestep = dt

    def set_state(self, x):
        self.x = x

    def get_traj(self):
        angle = np.pi / 6
        total_time_steps = int(40 / self.timestep)
        xcl = [
            np.zeros(
                X_DIM,
            )
        ]
        ucl = []
        u = np.zeros(
            U_DIM,
        )
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
            xcl.append(kinetic_bicycle(deepcopy(xcl[-1]), deepcopy(u), self.timestep))
            ucl.append(deepcopy(u))
        xcl, ucl = np.array(xcl).T[:, :-1], np.array(ucl).T[:, :-1]
        np.savetxt("data/closed_loop_feasible.txt", xcl, fmt="%f")
        self.xcl = xcl
        self.ucl = ucl

    def set_ctrl_policy(self, ctrl_policy):
        self.ctrl_policy = ctrl_policy

    def calc_ctrl_input(self):
        self.ctrl_policy.set_state(self.x)
        self.ctrl_policy.calc_input()
        self.u = self.ctrl_policy.get_input()

    def forward_one_step(self):
        self.calc_ctrl_input()
        self.forward_dynamics()
        self.ctrl_policy.set_state(self.x)
        self.update_memory()

    def update_memory(self):
        self.times.append(deepcopy(self.time))
        self.xs.append(deepcopy(self.x))
        self.inputs.append(deepcopy(self.u))

    def update_memory_post_iter(self):
        self.all_times.append(deepcopy(self.times))
        self.all_xs.append(deepcopy(self.xs))
        self.all_inputs.append(deepcopy(self.inputs))
        self.times = []
        self.inputs = []
        self.xs = []
        self.xs.append(np.zeros(X_DIM).tolist())
        self.set_state(np.zeros(X_DIM).tolist())

    def forward_dynamics(self):
        # This function computes the system evolution. Note that the discretization is delta_t and therefore is needed that
        # dt <= delta_t and ( dt / delta_t) = integer value
        # Discretization Parameters
        delta_t = 0.001
        x_next = np.zeros((X_DIM))
        x_next = self.x
        # Initialize counter
        # i = 0
        # while (i + 1) * delta_t <= self.timestep:
        #     if self.u is None:
        #         pass
        #     else:
        #         x_next = kinetic_bicycle(x_next, self.u, delta_t)
        #     # Increment counter
        #     i = i + 1
        x_next = kinetic_bicycle(x_next, self.u, self.timestep)
        # Noises
        if not self.zero_noise_flag:
            noise_v = np.maximum(-0.05, np.minimum(np.random.randn() * 0.01, 0.05))
            noise_theta = np.maximum(-0.05, np.minimum(np.random.randn() * 0.005, 0.05))
            x_next[X_ID["v"]] = x_next[X_ID["v"]] + 0.5 * noise_v
            x_next[X_ID["theta"]] = x_next[X_ID["theta"]] + 0.5 * noise_theta
        self.x = x_next.tolist()
        self.time += self.timestep


class ControlBase:
    def __init__(self):
        self.time = 0.0
        self.timestep = None
        self.x = None
        self.u = None
        # store the information (e.g. states, inputs) of current lap
        self.all_times, self.all_xs, self.all_inputs = [], [], []
        self.times, self.xs, self.inputs = [], [], []
        self.xs.append(np.zeros(X_DIM))
        self.iters = 0
        self.initial_traj_x = None
        self.initial_traj_u = None
        self.solver_times, self.solver_times = [], []

    def set_initial_traj(self, xcl, ucl):
        self.initial_traj_x = xcl
        self.initial_traj_u = ucl

    def set_timestep(self, timestep):
        self.timestep = timestep

    def set_state(self, x):
        self.x = x

    def calc_input(self):
        pass

    def get_input(self):
        return self.u

    def update_memory(self):
        self.times.append(deepcopy(self.time))
        self.xs.append(deepcopy(self.x))
        self.inputs.append(deepcopy(self.u))

    def update_memory_post_iter(self):
        self.all_times.append(deepcopy(self.times))
        self.all_xs.append(deepcopy(self.xs))
        self.all_inputs.append(deepcopy(self.inputs))
        self.times = []
        self.inputs = []
        self.xs = []
        self.xs.append(np.zeros(X_DIM))


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
    def __init__(self, lmpc_param, obstacle, system_param=None):
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
                    self.x,
                    self.x_guess,
                    x_terminal,
                    self.x_sol,
                    self.u_sol,
                    self.timestep,
                    self.num_horizon,
                    self.old_cost,
                    cost_terminal,
                    self.system_param,
                    self.obstacle
                )
                cost_iter.append(deepcopy(cost))
                input_iter.append(deepcopy(self.u_sol[:, 0]))
                x_pred_iter.append(deepcopy(self.x_sol))
                u_pred_iter.append(deepcopy(self.u_sol))
            id_list.append(deepcopy(index_ss_points))
            cost_list.append(deepcopy(cost_iter))
            u_list.append(deepcopy(input_iter))
            x_pred.append(deepcopy(x_pred_iter))
            u_pred.append(deepcopy(u_pred_iter))
        best_iter_loc_ss = cost_list.index(min(cost_list))
        cost_vec = cost_list[best_iter_loc_ss]
        best_time = cost_vec.index(min(cost_vec))
        best_iter = best_iter_loc_ss + min_iter
        self.u = u_list[best_iter_loc_ss][best_time]
        self.x_pred = x_pred[best_iter_loc_ss][best_time]
        self.u_pred = u_pred[best_iter_loc_ss][best_time]
        self.cost = cost_list[best_iter_loc_ss][best_time]
        if self.old_cost <= self.cost:
            print("ERROR: The cost is not decreasing")
            pdb.set_trace()
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
            # print("index list", id_list[best_iter_loc_ss][best_time])
            # print("ss", self.ss[best_iter].shape[1])
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
        self.ss.append(deepcopy(x))
        self.u_ss.append(deepcopy(u))
        self.Qfun.append(deepcopy(np.arange(x.shape[1] - 1, -1, -1)))
        self.num_horizon = self.lmpc_param.num_horizon
        self.x_terminal_guess = x[:, self.num_horizon]
        self.x_guess = np.concatenate(
            (
                x[:, 0 : (self.num_horizon + 1)].T.flatten(),
                u[:, 0 : (self.num_horizon)].T.flatten(),
            ),
            axis=0,
        )
        self.cost = self.Qfun[-1][0]
        self.iter_cost.append(deepcopy(self.cost))
        self.old_cost = self.cost + 1
        self.old_iter = self.iter
        self.x_sol = x[:, 0 : (self.num_horizon + 1)]
        self.u_sol = u[:, 0 : (self.num_horizon)]
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

    def sim(self, iter, sim_time=50.0):
        for i in range(0, int(sim_time / self.timestep)):
            # update system state
            self.robotic.forward_one_step()
            if np.linalg.norm(self.robotic.x - self.initial_traj[:, -1]) <= 1e-4:
                self.robotic.update_memory_post_iter()
                print("iteration: {}".format(iter) + " finished")
                break
    
    def plot_inputs(self):
        list_inputs = []
        list_times = []
        time = 0
        for iter in range(len(self.robotic.all_inputs)):
            for index in range(len(self.robotic.all_inputs[iter])):
                list_inputs.append(self.robotic.all_inputs[iter][index])
                list_times.append(time)
                time += self.timestep
        fix, axs = plt.subplots(2)
        u = np.asarray(list_inputs)
        axs[0].plot(list_times, u[:, 0], "-o", linewidth=1, markersize=1)
        axs[0].set_xlabel("time [s]", fontsize=14)
        axs[0].set_ylabel("$a$ [m/s^2]", fontsize=14)
        axs[1].plot(list_times, u[:, 1], "-o", linewidth=1, markersize=1)
        axs[1].set_xlabel("time [s]", fontsize=14)
        axs[1].set_ylabel("$/delta$ [rad]", fontsize=14)
        plt.show()
    
    def plot_simulation(self):
        list_xs = []
        for index in range(len(self.robotic.all_xs[-1])):
            list_xs.append(self.robotic.all_xs[-1][index])
        array_xs = np.asarray(list_xs)
        fig, ax = plt.subplots()
        self.robotic.ctrl_policy.obstacle.plot_obstacle()
        line1, = ax.plot(array_xs[:, 0], array_xs[:, 1], label="trajectory at last iteration")
        line2, = ax.plot(self.robotic.xcl[0, :], self.robotic.xcl[1,:], label="initial trajectory")
        plt.legend(handles=[line1,line2])
        plt.show()
        
        
        
        
