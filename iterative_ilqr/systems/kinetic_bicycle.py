import numpy as np
from utils.constants_kinetic_bicycle import *

# Add lambda functions
cos = lambda a : np.cos(a)
sin = lambda a : np.sin(a)
tan = lambda a : np.tan(a)


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


def get_A_matrix(velocity_vals, theta, acceleration_vals, num_horizon, dt):
    z = np.zeros((num_horizon))
    o = np.ones((num_horizon))
    v = velocity_vals
    v_dot = acceleration_vals
    A = np.array([[o, z, cos(theta)*dt, -(v*dt + (v_dot*dt**2)/2)*sin(theta)],
                      [z, o, sin(theta)*dt,  (v*dt + (v_dot*dt**2)/2)*cos(theta)],
                      [z, z,             o,                                         z],
                      [z, z,             z,                                         o]])
    return A

def get_B_matrix(theta, num_horizon, dt):
    z = np.zeros((num_horizon))
    o = np.ones((num_horizon))
    B = np.array([[dt**2*cos(theta)/2,         z],
                    [dt**2*sin(theta)/2,       z],
                    [dt*o,                     z],
                    [z,                     dt*o]])
    return B
