import pickle as pkl

import os
import numpy as np
from utils import base
from utils.constants_kinetic_bicycle import *
from copy import deepcopy


def test_ilqr(args):
    if args["direct_ilqr"]:
        direct_ilqr = True
    else:
        direct_ilqr = False
    num_horizon = 6
    dt = 1
    sim_time = 50
    lap_number = args["lap_number"]
    num_ss_iter = args["num_ss_iters"]
    num_ss_points = args["num_ss_points"]
    all_ss_point = False
    all_ss_iter = False
    x0 = [0, 0, 0, 0]
    ego = base.KineticBicycle(direct_ilqr=direct_ilqr, system_param=base.KineticBicycleParam())
    ego.set_state(x0)
    ego.set_timestep(dt)
    ego.get_traj()
    ego.set_zero_noise()

    ilqr_param = base.iLqrParam(
        num_ss_points=num_ss_points,
        num_ss_iter=num_ss_iter,
        timestep=dt,
        num_horizon=num_horizon,
        all_ss_iter=all_ss_iter,
        all_ss_point=all_ss_point,
    )
    ilqr = base.iLqr(ilqr_param, system_param=base.KineticBicycleParam())
    ilqr.add_trajectory(ego.xcl, ego.ucl)
    ilqr.set_timestep(dt)
    ego.set_ctrl_policy(ilqr)
    simulator = base.Simulator()
    simulator.set_robotic(ego)
    simulator.set_timestep(dt)
    simulator.set_traj()
    for iter in range(lap_number):
        print("iteration ", iter, "begins")
        simulator.sim(iter, sim_time=sim_time)
        ego_state = deepcopy(ego.data["state"][-1])
        ego_state[-1, :] = ego.xcl[-1, :]
        ilqr.add_trajectory(ego_state, ego.data["input"][-1])
    print("time at iteration 0 is", len(ego.xcl) * dt, " s")
    for id in range(len(ego.data["timestamp"])):
        lap = id + 1
        print("time at iteration ", lap, " is ", (len(ego.data["timestamp"][id]) * dt), " s")
    if args["plotting"]:
        simulator.plot_inputs()
        simulator.plot_simulation()
    with open("data/ego_ilqr_ss_" + str(num_ss_points) + "_no_obstacle.obj", "wb") as handle:
        pkl.dump(ego, handle, protocol=pkl.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--lap-number", type=int)
    parser.add_argument("--num-ss-points", type=int)
    parser.add_argument("--num-ss-iters", type=int)
    parser.add_argument("--plotting", action="store_true")
    parser.add_argument("--direct-ilqr", action="store_true")
    parser.add_argument("--save-trajectory", action="store_true")
    args = vars(parser.parse_args())
    test_ilqr(args)
