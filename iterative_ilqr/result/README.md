# Reuslts
This folder contains the code to run all the results in the paper:

## Iterative Tasks In Static Environments

#### No Obstacle(I2LQR)
Run
```
python iterative_ilqr/result/ilqr_test_no_obstacle.py --lap-number 10 --num-ss-iters 2 --num-ss-points 8 --plotting
```

This allows to test the iterative ilqr controller. The argparse arguments are listed as follow,

| name | type | choices | description |
| :---: | :---: | :---: | :---: |
| `lap_number` | int | any number that is greater than `2` | number of laps that will be simulated |
| `num_ss_iters` | int | any number that is greater than `1` | iterations used for learning |
| `num_ss_points` | int | any number that is greater than `1` | history states used for learning |
|   `plotting`   | action |               `store_true`                |                    save plotting if true                     |

#### No Obstacle(LMPC)
Run
```
python iterative_ilqr/result/nlmpc_test_no_obstacle.py --lap-number 10 --num-ss-iters 2 --num-ss-points 8 --ss-option space --plotting
```
This allows to test the nonlinear lmpc controller. The argparse arguments are listed as follow,
| name | type | choices | description |
| :---: | :---: | :---: | :---: |
| `lap_number` | int | any number that is greater than `2` | number of laps that will be simulated |
| `num_ss_iters` | int | any number that is greater than `1` | iterations used for learning |
| `num_ss_points` | int | any number that is greater than `1` | history states used for learning |
| `ss_option` | string | `space`, `time` or `all` | criteria for history states selection |
|   `plotting`   | action |               `store_true`                |                    save plotting if true                     |

#### Static Obstacle(I2LQR)
Run
```
python iterative_ilqr/result/ilqr_test_static_obstacle.py --lap-number 10 --num-ss-iters 2 --num-ss-points 8 --plotting
```

This allows to test the iterative ilqr controller. The argparse arguments are listed as follow,

| name | type | choices | description |
| :---: | :---: | :---: | :---: |
| `lap_number` | int | any number that is greater than `2` | number of laps that will be simulated |
| `num_ss_iters` | int | any number that is greater than `1` | iterations used for learning |
| `num_ss_points` | int | any number that is greater than `1` | history states used for learning |
|   `plotting`   | action |               `store_true`                |                    save plotting if true                     |

#### Static Obstacle(LMPC)
Run
```
python iterative_ilqr/result/nlmpc_test_static_obstacle.py --lap-number 10 --num-ss-iters 2 --num-ss-points 8 --ss-option space --plotting
```
This allows to test the nonlinear lmpc controller. The argparse arguments are listed as follow,
| name | type | choices | description |
| :---: | :---: | :---: | :---: |
| `lap_number` | int | any number that is greater than `2` | number of laps that will be simulated |
| `num_ss_iters` | int | any number that is greater than `1` | iterations used for learning |
| `num_ss_points` | int | any number that is greater than `1` | history states used for learning |
| `ss_option` | string | `space`, `time` or `all` | criteria for history states selection |
|   `plotting`   | action |               `store_true`                |                    save plotting if true                     |

## Iterative Tasks In Dynamic Environments

#### Add Static Obstacle(I2LQR)
Run
```
python iterative_ilqr/result/ilqr_test_add_static_obstacle.py --lap-number 6 --num-ss-iters 2 --num-ss-points 8 --plotting
```

This allows to test the iterative ilqr controller. The argparse arguments are listed as follow,

| name | type | choices | description |
| :---: | :---: | :---: | :---: |
| `lap_number` | int | any number that is greater than `2` | number of laps that will be simulated |
| `num_ss_iters` | int | any number that is greater than `1` | iterations used for learning |
| `num_ss_points` | int | any number that is greater than `1` | history states used for learning |
|   `plotting`   | action |               `store_true`                |                    save plotting if true                     |

#### Add Static Obstacle(LMPC)
Run
```
python iterative_ilqr/result/nlmpc_test_add_static_obstacle.py --lap-number 6 --num-ss-iters 2 --num-ss-points 8 --ss-option space --plotting
```
This allows to test the nonlinear lmpc controller. The argparse arguments are listed as follow,
| name | type | choices | description |
| :---: | :---: | :---: | :---: |
| `lap_number` | int | any number that is greater than `2` | number of laps that will be simulated |
| `num_ss_iters` | int | any number that is greater than `1` | iterations used for learning |
| `num_ss_points` | int | any number that is greater than `1` | history states used for learning |
| `ss_option` | string | `space`, `time` or `all` | criteria for history states selection |
|   `plotting`   | action |               `store_true`                |                    save plotting if true                     |

#### Add Moving Obstacle(I2LQR)
Run
```
python iterative_ilqr/result/ilqr_test_add_moving_obstacle.py --lap-number 6 --num-ss-iters 2 --num-ss-points 8 --plotting --moving-option up
```

This allows to test the iterative ilqr controller. The argparse arguments are listed as follow,

| name | type | choices | description |
| :---: | :---: | :---: | :---: |
| `lap_number` | int | any number that is greater than `2` | number of laps that will be simulated |
| `num_ss_iters` | int | any number that is greater than `1` | iterations used for learning |
| `num_ss_points` | int | any number that is greater than `1` | history states used for learning |
| `moving-option` | string | `up`, `left` | criteria for obstacle selection |
|   `plotting`   | action |               `store_true`                |                    save plotting if true                     |

#### Add Moving Obstacle(LMPC)
Run
```
python iterative_ilqr/result/nlmpc_test_add_moving_obstacle.py --lap-number 6 --num-ss-iters 2 --num-ss-points 8 --ss-option space --plotting --moving-option up
```
This allows to test the nonlinear lmpc controller. The argparse arguments are listed as follow,
| name | type | choices | description |
| :---: | :---: | :---: | :---: |
| `lap_number` | int | any number that is greater than `2` | number of laps that will be simulated |
| `num_ss_iters` | int | any number that is greater than `1` | iterations used for learning |
| `num_ss_points` | int | any number that is greater than `1` | history states used for learning |
| `ss_option` | string | `space`, `time` or `all` | criteria for history states selection |
| `moving-option` | string | `up`, `left` | criteria for obstacle selection |
|   `plotting`   | action |               `store_true`                |                    save plotting if true                     |