import os
import numpy as np
from utils import base
from utils.constants_kinetic_bicycle import *
from copy import deepcopy


def test_ilqr(args):
    if args["save_trajectory"]:
        save_ilqr_traj = True
    else:
        save_ilqr_traj = False
    num_horizon = 6
    dt = 1
    sim_time = 50
    ss_optioin = "spaceVarying"
    lap_number = args["lap_number"]
    num_ss_iter = args["num_ss_iters"]
    num_ss_points = args["num_ss_points"]
    all_ss_point = False
    all_ss_iter = False
    x0 = np.zeros((X_DIM,))
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
    ilqr_param = base.iLqrParam(
        num_ss_points=num_ss_points,
        num_ss_iter=num_ss_iter,
        timestep=dt,
        ss_option=ss_optioin,
        num_horizon=num_horizon,
        all_ss_iter=all_ss_iter,
        all_ss_point=all_ss_point,
    )
    ilqr = base.iLqr(ilqr_param, obstacle=obstacle, system_param=base.KineticBicycleParam())
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
        ego.data["state"][-1] = np.vstack((ego.data["state"][-1], ego.xcl[-1,:]))
        ilqr.add_trajectory(ego.data["state"][-1], ego.data["input"][-1])
    print("time at iteration 0 is", len(ego.xcl) * dt, " s")
    for id in range(len(ego.data["timestamp"])):
        lap = id + 1
        print("time at iteration ", lap, " is ", (len(ego.data["timestamp"][id]) * dt), " s")
    if args["plotting"]:
        simulator.plot_inputs()
        simulator.plot_simulation()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lap-number", type=int)
    parser.add_argument("--num-ss-points", type=int)
    parser.add_argument("--num-ss-iters", type=int)
    parser.add_argument("--plotting", action="store_true")
    parser.add_argument("--save-trajectory", action="store_true")
    args = vars(parser.parse_args())
    test_ilqr(args)