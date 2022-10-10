import numpy as np
from utils.constants_kinetic_bicycle import *

def kinetic_bicycle(x,u,dt):
    '''
    Kinematic bicycle model
    state: x, y, speed (v), heading angle (theta)
    input: accleration (accel), steering angle (delta)
    dt: sampling time
    '''
    x_next = np.array([x[X_ID["x"]] + np.cos(x[X_ID["theta"]])*(x[X_ID["v"]]*dt + (u[U_ID["accel"]]*dt**2)/2),
                    x[X_ID["y"]] + np.sin(x[X_ID["theta"]])*(x[X_ID["v"]]*dt + (u[U_ID["accel"]]*dt**2)/2),
                    x[X_ID["v"]] + u[U_ID["accel"]]*dt,
                    x[X_ID["theta"]] + u[U_ID["delta"]]*dt])
    return x_next