import numpy as np
from copy import deepcopy

class KineticBicycleParam:
    def __init__(self, delta_max=np.pi/2, a_max=2.0, v_max=10, v_min=0):
        self.delta_max = delta_max
        self.a_max = a_max
        self.v_max = v_max
        self.v_min = v_min


class KineticBicycle:
    def __init__(self, name=None, system_param=None):
        self.system_param = system_param
        self.time = 0.0
        self.timestep = None
        self.x = None
        self.u = None
        self.zero_noise_flag = False
        self.all_times, self.all_xs, self.all_inputs = [], [], []
        self.times, self.xs, self.inputs = [], [], []
        self.iters = 0
        self.solver_times, self.solver_times = [], []

    def set_zero_noise(self):
        self.zero_noise_flag = True

    def set_timestep(self, dt):
        self.timestep = dt

    def set_state(self, x):
        self.x = x

    def get_traj(self):
        angle = np.pi/6
        total_time_steps = 40
        xcl = [np.zeros(X_DIM,)]
        ucl =[]
        u = np.zeros(U_DIM,)

        # Simple brute force hard coded if logic
        for i in range(0, total_time_steps):
            if i <= 1:
                u["accel"] = 1
            elif i==(total_time_steps-4) or i==(total_time_steps-3):
                u["accel"] =  -1
            else:
                u["accel"] = 0   

            if i == 1:
                u["delta"] =  angle
            elif i == int(total_time_steps/2)-2 or i== int(total_time_steps/2)-1: 
                u["delta"] = -angle
            else:
                u["delta"] = 0
            xcl.append(kinetic_bicycle(xcl[-1], u, self.timestep))
            ucl.append(u)
        np.savetxt('data/closed_loop_feasible.txt',xcl, fmt='%f' )
        self.xcl = xcl
        self.ucl = ucl


    def set_ctrl_policy(self, ctrl_policy):
        self.ctrl_policy = ctrl_policy
        self.ctrl_policy.agent_name = self.name

    def calc_ctrl_input(self):
        self.ctrl_policy.set_state(self.xcurv, self.xglob)
        self.ctrl_policy.calc_input()
        self.u = self.ctrl_policy.get_input()

    def forward_one_step(self):
        self.calc_ctrl_input()
        self.forward_dynamics()
        self.ctrl_policy.set_state(self.xcurv, self.xglob)
        self.update_memory()

    def update_memory(self):
        self.times.append(deepcopy(self.time))
        self.xs.append(deepcopy(self.x))
        self.inputs.append(deepcopu(self.u))
    
    def update_memory_post_iter(self):
        self.all_times.append(deepcopy(self.times))
        self.all_xs.append(deepcopy(self.xs))
        self.all_inputs.append(deepcopy(self.inputs))
        self.times = []
        self.inputs = []
        self.xs = []

     def forward_dynamics(self):
        # This function computes the system evolution. Note that the discretization is delta_t and therefore is needed that
        # dt <= delta_t and ( dt / delta_t) = integer value
        # Discretization Parameters
        delta_t = 0.001
        xglob_next = np.zeros((X_DIM,))
        xcurv_next = np.zeros((X_DIM,))
        xglob_next = self.xglob
        xcurv_next = self.xcurv
        vehicle_param = CarParam()
        # Initialize counter
        i = 0
        while (i + 1) * delta_t <= self.timestep:
            s = xcurv_next[XCURV_ID["s"]]
            curv = self.track.get_curvature(s)
            if self.u is None:
                pass
            else:
                xglob_next, xcurv_next = vehicle_dynamics.vehicle_dynamics(
                    vehicle_param.dynamics_param,
                    curv,
                    xglob_next,
                    xcurv_next,
                    delta_t,
                    self.u,
                )
            # Increment counter
            i = i + 1
        # Noises
        noise_vx = np.maximum(-0.05, np.minimum(np.random.randn() * 0.01, 0.05))
        noise_vy = np.maximum(-0.1, np.minimum(np.random.randn() * 0.01, 0.1))
        noise_wz = np.maximum(-0.05, np.minimum(np.random.randn() * 0.005, 0.05))
        xcurv_next[XCURV_ID["vx"]] = xcurv_next[XCURV_ID["vx"]] + 0.5 * noise_vx
        xcurv_next[XCURV_ID["vy"]] = xcurv_next[XCURV_ID["vy"]] + 0.5 * noise_vy
        xcurv_next[XCURV_ID["wz"]] = xcurv_next[XCURV_ID["wz"]] + 0.5 * noise_wz
        self.xcurv = xcurv_next
        self.xglob = xglob_next
        self.time += self.timestep





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
        self.initial_traj = self.robotic.get_traj()

    def sim(self, iter, sim_time=50.0):
        for i in range(0, int(sim_time / self.timestep)):
            # update system state
            self.robotic.forward_one_step()
            if np.linalg.norm(self.robotic.x - initial_traj[:,-1]) <= 1e-4:
                self.robotic.update_memory_post_iter()
                print("iteration: {}".format(iter)+" finished")
                break
