from utils.constants_kinetic_bicycle import *
from systems.kinetic_bicycle import *
from casadi import *


def nlmpc(
    x,
    x_guess,
    x_terminal,
    x_sol_old,
    u_sol_old,
    timestep,
    num_horizon,
    old_cost,
    cost_terminal,
    sys_param,
    obstacle
):
    if num_horizon > 1:
        if num_horizon + cost_terminal < old_cost:
            X = SX.sym("X", X_DIM * (num_horizon + 1))
            U = SX.sym("X", U_DIM * num_horizon)
            slack = SX.sym("X", X_DIM)
            slack_obs = SX.sym("X", (num_horizon - 1))
            constraint = []
            for i in range(0, num_horizon):
                constraint = vertcat(
                    constraint,
                    X[X_DIM * (i + 1) + 0]
                    - (
                        X[X_DIM * i + 0]
                        + np.cos(X[X_DIM * i + 3])
                        * (X[X_DIM * i + 2] * timestep + (U[U_DIM * i + 0] * timestep**2) / 2)
                    ),
                )
                constraint = vertcat(
                    constraint,
                    X[X_DIM * (i + 1) + 1]
                    - (
                        X[X_DIM * i + 1]
                        + np.sin(X[X_DIM * i + 3])
                        * (X[X_DIM * i + 2] * timestep + (U[U_DIM * i + 0] * timestep**2) / 2)
                    ),
                )
                constraint = vertcat(
                    constraint,
                    X[X_DIM * (i + 1) + 2] - (X[X_DIM * i + 2] + U[U_DIM * i + 0] * timestep),
                )
                constraint = vertcat(
                    constraint,
                    X[X_DIM * (i + 1) + 3] - (X[X_DIM * i + 3] + U[U_DIM * i + 1] * timestep),
                )
            for i in range(1, num_horizon):
                constraint = vertcat(
                    constraint,
                    ((X[X_DIM * i + 0] - obstacle.x) ** 2 / (obstacle.width**2))
                    + ((X[X_DIM * i + 1] - obstacle.y) ** 2 / (obstacle.height**2))
                    - slack_obs[i - 1],
                )               
            constraint = vertcat(
                constraint,
                X[X_DIM * num_horizon : X_DIM * (num_horizon + 1)] - x_terminal + slack,
            )
            cost = 1000000 * (slack[0] ** 2 + slack[1] ** 2 + slack[2] ** 2 + slack[3] ** 2)
            for i in range(0, num_horizon):
                cost = cost + 1
            opts = {
                "verbose": False,
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.mu_strategy": "adaptive",
                "ipopt.mu_init": 1e-5,
                "ipopt.mu_min": 1e-15,
                "ipopt.barrier_tol_factor": 1,
            }
            nlp = {"x": vertcat(X, U, slack, slack_obs), "f": cost, "g": constraint}
            solver = nlpsol("solver", "ipopt", nlp, opts)
            lbg_dyanmics = [0] * (X_DIM * num_horizon) + [0 * 1.0] * (num_horizon - 1) + [0] * X_DIM
            ubg_dyanmics = (
                [0] * (X_DIM * num_horizon) + [0 * 100000000] * (num_horizon - 1) + [0] * X_DIM
            )
            lbx = (
                x
                + [-1000] * (X_DIM * (num_horizon))
                + [-sys_param.a_max, -sys_param.delta_max] * num_horizon
                + [-1000] * X_DIM
                + [1] * (num_horizon - 1)
            )
            ubx = (
                x
                + [1000] * (X_DIM * (num_horizon))
                + [sys_param.a_max, sys_param.delta_max] * num_horizon
                + [1000] * X_DIM
                + [100000] * (num_horizon - 1)
            )
            xGuessTot = np.concatenate((x_guess, np.zeros(X_DIM + num_horizon - 1)), axis=0)
            sol = solver(
                lbx=lbx,
                ubx=ubx,
                lbg=lbg_dyanmics,
                ubg=ubg_dyanmics,
                x0=xGuessTot.tolist(),
            )
            x = np.array(sol["x"])
            x_sol = x[0 : (num_horizon + 1) * X_DIM].reshape((num_horizon + 1, X_DIM)).T
            u_sol = (
                x[(num_horizon + 1) * X_DIM : ((num_horizon + 1) * X_DIM + U_DIM * num_horizon)]
                .reshape((num_horizon, U_DIM))
                .T
            )
            slack_sol = x[
                ((num_horizon + 1) * X_DIM + U_DIM * num_horizon) : (
                    (num_horizon + 1) * X_DIM + U_DIM * num_horizon + X_DIM
                )
            ]
            if (solver.stats()["success"]) and (np.linalg.norm(slack_sol, 2) < 1e-8):
                feasible = 1
            else:
                feasible = 0
            cost = num_horizon + cost_terminal if feasible else float("Inf")
        else:
            cost = 100
            x_sol = x_sol_old
            u_sol = u_sol_old
    else:
        xNext = kinetic_bicycle(
            x,
            x_guess[X_DIM * (num_horizon + 1) : (X_DIM * (num_horizon + 1) + U_DIM)],
            timestep,
        )
        if np.linalg.norm([xNext - x_terminal]) <= 1e-3:
            cost = 1 + cost_terminal
            x_sol = np.vstack((x, x_terminal)).T
            u_sol = np.zeros((U_DIM, 1))
            u_sol[:, 0] = x_guess[X_DIM * (num_horizon + 1) : (X_DIM * (num_horizon + 1) + U_DIM)]
        else:
            cost = float("Inf")
            x_sol = float("Inf") * np.vstack((x, x_terminal)).T
            u_sol = float("Inf") * np.ones((U_DIM, 1))
    return x_sol, u_sol, cost
