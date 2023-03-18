import pytest
import numpy as np
from utils import base
from utils.constants_kinetic_bicycle import *
from copy import deepcopy


@pytest.mark.parametrize("ss_option", ["space"])
def test_ilqr(ss_option):
    num_horizon = 6
    dt = 1
    sim_time = 50
    lap_number = 5
    num_ss_iter = 1
    num_ss_points = 8
    all_ss_point = False
    all_ss_iter = False
    if ss_option == "space":
        ss_optioin = "spaceVarying"
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
    ilqr = base.iLqr(ilqr_param, obstacle, system_param=base.KineticBicycleParam())
    ilqr.add_trajectory(ego.xcl, ego.ucl)
    ilqr.set_timestep(dt)
    ilqr.set_state(x0)
    ego.set_ctrl_policy(ilqr)
    simulator = base.Simulator()
    simulator.set_robotic(ego)
    simulator.set_timestep(dt)
    simulator.set_traj()
    for iter in range(lap_number):
        print("iteration ", iter, "begins")
        simulator.sim(iter, sim_time=sim_time)
        ego.data["state"][-1] = np.vstack((ego.data["state"][-1], ego.xcl[-1, :]))
        ilqr.add_trajectory(ego.data["state"][-1], ego.data["input"][-1])
    print("time at iteration 0 is", len(ego.xcl) * dt, " s")
    for id in range(len(ego.data["timestamp"])):
        lap = id + 1
        print("time at iteration ", lap, " is ", (len(ego.data["timestamp"][id]) * dt), " s")
    simulator.plot_inputs()
    simulator.plot_simulation()
