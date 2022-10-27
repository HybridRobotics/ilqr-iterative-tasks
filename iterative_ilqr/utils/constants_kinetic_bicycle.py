# System state / input dimensions
X_DIM, U_DIM = 4, 2

# System state / input descriptions
X_ID = {"x": 0, "y": 1, "v": 2, "theta": 3}
U_ID = {"accel": 0, "delta": 1}

# System state / input constriants
ACCEL_MAX, ACCEL_MIN = 2.0, -2.0
DELTA_MAX, DELTA_MIN = 1.57, -1.57
VELOCITY_MAX, VELOCITY_MIN = 3.14/2, -3.14/2
