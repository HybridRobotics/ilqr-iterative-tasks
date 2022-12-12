import pickle as pkl
import matplotlib.pyplot as plt
from utils.constants_kinetic_bicycle import *
from utils import base
import numpy as np

num_point_ilqr = [8,10,20]
num_point_lmpc = [8,10,20]

plt.figure(figsize=(8, 4))

colorMap = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# count = 0
# with open("data/ego_nlmpc_ss_"+str(20)+"_static_obstacle.obj", "rb") as handle:
#     ego = pkl.load(handle)
# plt.plot(ego.xcl[:,X_ID["x"]], ego.xcl[:,X_ID["y"]], '-o', color="black")

# obstacle = ego.ctrl_policy.obstacle

# x_obs = obstacle.x0
# y_obs = obstacle.y0
# width_obs = obstacle.width
# height_obs = obstacle.height

# x_obs_list = []
# y_obs_list = []
# for index in np.linspace(0,2*np.pi,1000):
#     x_obs_list.append(x_obs + width_obs*np.cos(index))
#     y_obs_list.append(y_obs + height_obs*np.sin(index))
# plt.plot(x_obs_list, y_obs_list, '-k',markersize=7, label="Obstacle", linewidth=2)


# lines = []
# for point in num_point_lmpc:
#     with open("data/ego_nlmpc_ss_"+str(point)+"_static_obstacle.obj", "rb") as handle:
#         ego = pkl.load(handle)
#     line0,=plt.plot(ego.data["state"][5][:,X_ID["x"]],ego.data["state"][9][:,X_ID["y"]],'-^',markersize=7, color=colorMap[count], label="LMPC, P="+str(point))
#     lines.append(line0)
#     count+=1
# for point in num_point_ilqr:
#     with open("data/ego_ilqr_ss_"+str(point)+"_static_obstacle.obj", "rb") as handle:
#         ego = pkl.load(handle)
#     line0,=plt.plot(ego.data["state"][5][:,X_ID["x"]],ego.data["state"][9][:,X_ID["y"]],'-D',markersize=7, color=colorMap[count],label="CDILQR, P="+str(point))
#     lines.append(line0)
#     count+=1
# plt.axis("equal")
# plt.legend(handles=lines)
# plt.xlabel("$p_x$")
# plt.ylabel("$p_y$")
# plt.tight_layout()
# plt.savefig("media/static_obstacle_trajectory.png",format="png")
# plt.show()

# ################################################

# plt.figure(figsize=(5, 2.5))

# colorMap = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# count = 0


# lines = []
# for point in num_point_lmpc:
#     cost = []
#     with open("data/ego_nlmpc_ss_"+str(point)+"_static_obstacle.obj", "rb") as handle:
#         ego = pkl.load(handle)
#     cost.append(len(ego.xcl))
#     for iter in range(10):
#         cost.append(len(ego.data["timestamp"][iter]))
#     line0,=plt.plot(cost,'-^',markersize=7, color=colorMap[count], label="LMPC, P="+str(point))
#     lines.append(line0)
#     count+=1

# for point in num_point_ilqr:
#     cost = []
#     cost.append(len(ego.xcl))
#     with open("data/ego_ilqr_ss_"+str(point)+"_static_obstacle.obj", "rb") as handle:
#         ego = pkl.load(handle)
#     for iter in range(10):
#         cost.append(len(ego.data["timestamp"][iter]))
#     line0,=plt.plot(cost,'-D',markersize=7, color=colorMap[count],label="CDILQR, P="+str(point))
#     lines.append(line0)
#     count+=1

# plt.legend(handles=lines)
# plt.xlabel("Iteration $j$")
# plt.ylabel("Timestamps")
# plt.tight_layout()
# plt.savefig("media/static_obstacle_time.png",format="png")
# plt.show()

# plt.figure(figsize=(5, 2.5))
fig, axs = plt.subplots(2)
fig.set_figheight(6)
fig.set_figwidth(6)

point = 10
with open("data/ego_nlmpc_ss_"+str(point)+"_static_obstacle.obj", "rb") as handle:
    ego = pkl.load(handle)
list_inputs_lmpc = (ego.data["input"][4])
list_states_lmpc = (ego.data["state"][4])
list_times = np.arange(0, len(list_inputs_lmpc))
axs[1].plot(list_times, list_inputs_lmpc[:, U_ID["accel"]], "-^", linewidth=1, markersize=1)
axs[1].set_xlabel("time [s]", fontsize=14)
axs[1].set_ylabel("$a$ [m/s^2]", fontsize=14)
list_times = np.arange(0, len(list_states_lmpc))
axs[0].plot(list_times, list_states_lmpc[:, X_ID["v"]], "-^", linewidth=1, markersize=1)
axs[0].set_xlabel("time [s]", fontsize=14)
axs[0].set_ylabel("$v$ [m/s]", fontsize=14)



point = 8
with open("data/ego_ilqr_ss_"+str(point)+"_static_obstacle.obj", "rb") as handle:
    ego = pkl.load(handle)
list_inputs_ilqr = (ego.data["input"][3])
list_states_ilqr = (ego.data["state"][3])
list_times = np.arange(0, len(list_inputs_ilqr))
axs[1].plot(list_times, list_inputs_ilqr[:, U_ID["accel"]], "-D", linewidth=1, markersize=1)
axs[1].set_xlabel("time [s]", fontsize=14)
axs[1].set_ylabel("$a$ [m/s^2]", fontsize=14)
list_times = np.arange(0, len(list_states_ilqr)-1)
axs[0].plot(list_times, list_states_ilqr[:-1, X_ID["v"]], "-D", linewidth=1, markersize=1)
axs[0].set_xlabel("time [s]", fontsize=14)
axs[0].set_ylabel("$v$ [m/s]", fontsize=14)




plt.show()

    