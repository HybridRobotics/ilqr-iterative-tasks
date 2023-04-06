import pickle as pkl

import numpy as np
from utils import base
from utils.constants_kinetic_bicycle import *


def nlmpc_test(args):
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
    x0 = np.zeros((X_DIM,))
    ego = base.KineticBicycle(system_param=base.KineticBicycleParam())
    ego.set_state(x0)
    ego.set_timestep(dt)
    ego.get_traj()
    ego.set_zero_noise()
    x_obs = 100
    y_obs = -5
    width_obs = 20
    height_obs = 40
    obstacle = base.Obstacle(x_obs, y_obs, width_obs, height_obs)
    obs_spd = 0
    obstacle = base.Obstacle(x_obs, y_obs, width_obs, height_obs, spd=obs_spd, timestep=dt)
    lmpc_param = base.LMPCParam(
        num_ss_points=num_ss_points,
        num_ss_iter=num_ss_iter,
        timestep=dt,
        ss_option=ss_optioin,
        num_horizon=num_horizon,
        all_ss_iter=all_ss_iter,
        all_ss_point=all_ss_point,
    )
    lmpc = base.LMPC(lmpc_param, obstacle=obstacle, system_param=base.KineticBicycleParam())
    lmpc.add_trajectory(ego.xcl, ego.ucl)
    lmpc.set_timestep(dt)
    ego.set_ctrl_policy(lmpc)
    simulator = base.Simulator()
    simulator.set_robotic(ego)
    simulator.set_timestep(dt)
    simulator.set_traj()
    for iter in range(lap_number):
        print("iteration ", iter, "begins")
        simulator.sim(iter, sim_time=sim_time)
        lmpc.add_trajectory(ego.data["state"][-1], ego.data["input"][-1])
    print("time at iteration 0 is", len(ego.xcl) * dt, " s")
    for id in range(len(ego.data["timestamp"])):
        lap = id + 1
        print("time at iteration ", lap, " is ", (len(ego.data["timestamp"][id]) * dt), " s")
    if args["plotting"]:
        simulator.plot_inputs()
        simulator.plot_simulation()
    with open("data/ego_nlmpc_ss_" + str(num_ss_points) + "_static_obstacle.obj", "wb") as handle:
        pkl.dump(ego, handle, protocol=pkl.HIGHEST_PROTOCOL)


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
