iterative-ilqr
==========

This repository provides a toolkit to test iterative learning controller.


## Features

## Installation
* We recommend creating a new conda environment:
```
conda env create -f environment.yml
conda activate iterative-ilqr
```

Run following command in terminal to install the iterative-ilqr package.
```
pip install -e .
```

## Auto Testing

In this project, `pytest` is used to test the code autonomously after pushing new code to the repository. Currently, three files in the `tests` folder are used for testing pid or mpc tracking controller, mpc-cbf controller and racing game planner, respectively. To test other features, add files to the `tests` folder and update the `tests.yml` file under the `.github/workflows` folder.

## Contributing
Execute `pre-commit install` to install git hooks in your `.git/` directory, which allows auto-formatting if you are willing to contribute to this repository.

Please contact major contributors of this repository for additional information.

## Quick-Demos

## Docs
The following documentation contains documentation and common terminal commands for simulations and testing.


#### Nonlinear LMPC
Run
```
python iterative_ilqr/tests/nlmpc_test.py --lap-number 5 --num-ss-iters 1 --num-ss-points 8 --ss-option space
```
This allows to test the nonlinear lmpc controller. The argparse arguments are listed as follow,
| name | type | choices | description |
| :---: | :---: | :---: | :---: |
| `lap_number` | int | any number that is greater than `2` | number of laps that will be simulated |
| `num_ss_iters` | int | any number that is greater than `1` | iterations used for learning |
| `num_ss_points` | int | any number that is greater than `1` | history states used for learning |
| `ss_option` | string | `space`, `time` or `all` | criteria for history states selection |
|   `plotting`   | action |               `store_true`                |                    save plotting if true                     |

#### Known Issues
To change the simulation timestep, the number of prediction horizon and number of history states used for learning should be changed.

No noise is added to the simulation during the dynamics update. The noise will result in failure when the robotics approaches the terminal point.

Current discretization time for system dynamics update is same as the simulation timestep. A smaller this value will also result in failure.



