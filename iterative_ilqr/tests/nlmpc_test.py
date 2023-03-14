import numpy as np
from utils import base
from utils.constants_kinetic_bicycle import *


def nlmpc_test(args):
    if args["save_trajectory"]:
        save_lmpc_traj = True
    else:
        save_lmpc_traj = False
    num_horizon = 6
    dt = 1
    sim_time = 50
    lap_number = args["lap_number"]
    num_ss_iter = args["num_ss_iters"]
    num_ss_points = args["num_ss_points"]
    if args["ss_option"] == "all":
        all_ss_point = True
        all_ss_iter = True
        ss_optioin = None
    else:
        all_ss_point = False
        all_ss_iter = False
        if args["ss_option"] == "space":
            ss_optioin = "spaceVarying"
        elif args["ss_option"] == "time":
            ss_optioin = "timeVarying"
    x0 = [0, 0, 0, 0]
    ego = base.KineticBicycle(system_param=base.KineticBicycleParam())
    ego.set_state(x0)
    ego.set_timestep(dt)
    ego.get_traj()
    ego.set_zero_noise()
    x_obs = 31
    y_obs = -2
    width_obs = 8
    height_obs = 6
    obstacle = base.Obstacle(x_obs, y_obs, width_obs, height_obs)
    lmpc_param = base.LMPCParam(
        num_ss_points=num_ss_points,
        num_ss_iter=num_ss_iter,
        timestep=dt,
        ss_option=ss_optioin,
        num_horizon=num_horizon,
        all_ss_iter=all_ss_iter,
        all_ss_point=all_ss_point,
    )
    lmpc = base.LMPC(lmpc_param, obstacle, system_param=base.KineticBicycleParam())
    lmpc.add_trajectory(ego.xcl, ego.ucl)
    lmpc.set_initial_traj(ego.xcl, ego.ucl)
    lmpc.set_timestep(dt)
    lmpc.set_state(x0)
    ego.set_ctrl_policy(lmpc)
    simulator = base.Simulator()
    simulator.set_robotic(ego)
    simulator.set_timestep(dt)
    simulator.set_traj()
    for iter in range(lap_number):
        print("iteration ", iter, "begins")
        simulator.sim(iter, sim_time=sim_time)
        lmpc.add_trajectory(np.array(ego.all_xs[-1]).T, np.array(ego.all_inputs[-1]).T)
    print("time at iteration 0 is", len(ego.xcl.T) * dt, " s")
    if save_lmpc_traj == True:
        np.savetxt('data/nlmpc_closed_loop_multi_laps.txt', np.round(np.array(ego.data["state"][-1]), decimals=5), fmt='%f' )
        np.savetxt('data/nlmpc_input_multi_laps.txt', np.round(np.array(ego.data["input"][-1]), decimals=5), fmt='%f' )
    for id in range(len(ego.all_times)):
        lap = id + 1
        print("time at iteration ", lap, " is ", (len(ego.all_times[id]) * dt), " s")
    if args["plotting"]:
        simulator.plot_inputs()
        simulator.plot_simulation()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lap-number", type=int)
    parser.add_argument("--num-ss-points", type=int)
    parser.add_argument("--num-ss-iters", type=int)
    parser.add_argument("--ss-option", type=str)
    parser.add_argument("--plotting", action="store_true")
    args = vars(parser.parse_args())
    nlmpc_test(args)
