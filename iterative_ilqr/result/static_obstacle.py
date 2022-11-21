import pickle as pkl
import matplotlib.pyplot as plt
from utils.constants_kinetic_bicycle import *


num_point_ilqr = [8,10,20]
num_point_lmpc = [8,10,20]



colorMap = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

count = 0
with open("data/ego_nlmpc_ss_"+str(20)+"_static_obstacle.obj", "rb") as handle:
    ego = pkl.load(handle)
plt.plot(ego.xcl[:,X_ID["x"]], ego.xcl[:,X_ID["y"]], color="black")
for point in num_point_lmpc:
    with open("data/ego_nlmpc_ss_"+str(point)+"_static_obstacle.obj", "rb") as handle:
        ego = pkl.load(handle)
    plt.plot(ego.data["state"][5][:,X_ID["x"]],ego.data["state"][5][:,X_ID["y"]], color=colorMap[count])
    count+=1
for point in num_point_ilqr:
    with open("data/ego_ilqr_ss_"+str(point)+"_static_obstacle.obj", "rb") as handle:
        ego = pkl.load(handle)
    plt.plot(ego.data["state"][5][:,X_ID["x"]],ego.data["state"][5][:,X_ID["y"]], color=colorMap[count])
    count+=1
plt.show()
    