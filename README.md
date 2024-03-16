iterative-ilqr-tasks
==========

This repository provides a toolkit to test a novel control strategy called Iterative Linear Quadratic Regulator for Iterative Tasks (i2LQR). The strategy aims to improve closed-loop performance with local trajectory optimization for iterative tasks in a dynamic environment.


## References
If you find this project useful in your work, please consider citing following paper [[arXiv]](https://arxiv.org/abs/2302.14246) | [[IEEE]](https://ieeexplore.ieee.org/document/10383960):
```
@inproceedings{zeng2023i2lqr,
  title={i2LQR: Iterative LQR for Iterative Tasks in Dynamic Environments},
  author={Zeng, Yifan and He, Suiyi and Nguyen, Han Hoang and Li, Yihan and Li, Zhongyu and Sreenath, Koushil and Zeng, Jun},
  booktitle={2023 62nd IEEE Conference on Decision and Control (CDC)},
  pages={5255--5260},
  year={2023},
  organization={IEEE}
}
```

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

In this project, `pytest` is used to test the code autonomously after pushing new code to the repository. Currently, files in the `tests` folder are used for testing i2LQR controller and nonlinear learning based MPC (NLMPC) controller, respectively. To test other features, add files to the `tests` folder and update the `tests.yml` file under the `.github/workflows` folder.

## Contributing
Execute `pre-commit install` to install git hooks in your `.git/` directory, which allows auto-formatting if you are willing to contribute to this repository.

Please contact major contributors of this repository for additional information.

## Quick-Demos

## Docs
The following documentation contains documentation and common terminal commands for simulations and testing.

#### Nonlinear LMPC
Run
```
python iterative_ilqr/tests/nlmpc_test.py --lap-number 10 --num-ss-iters 2 --num-ss-points 8 --ss-option space
```
This allows to test the nonlinear lmpc controller. The argparse arguments are listed as follow,
| name | type | choices | description |
| :---: | :---: | :---: | :---: |
| `lap_number` | int | any number that is greater than `2` | number of laps that will be simulated |
| `num_ss_iters` | int | any number that is greater than `1` | iterations used for learning |
| `num_ss_points` | int | any number that is greater than `1` | history states used for learning |
| `ss_option` | string | `space`, `time` or `all` | criteria for history states selection |
|   `plotting`   | action |               `store_true`                |                    save plotting if true                     |
|   `save_trajectory`   | action |               `store_true`                |                    save simulator will store the history states and inputs if true                     |


#### Iterative lqr for iterative tasks
Run
```
python iterative_ilqr/tests/ilqr_test.py --lap-number 10 --num-ss-iters 2 --num-ss-points 8
```

This allows to test the iterative ilqr controller. The argparse arguments are listed as follow,

| name | type | choices | description |
| :---: | :---: | :---: | :---: |
| `lap_number` | int | any number that is greater than `2` | number of laps that will be simulated |
| `num_ss_iters` | int | any number that is greater than `1` | iterations used for learning |
| `num_ss_points` | int | any number that is greater than `1` | history states used for learning |
|   `plotting`   | action |               `store_true`                |                    save plotting if true                     |
|   `save_trajectory`   | action |               `store_true`                |                    save simulator will store the history states and inputs if true                     |

#### Known Issues
- To change the simulation timestep, the number of prediction horizons and the number of history states used for learning should be adjusted.
- No noise is added to the simulation during the dynamics update. The presence of noise may lead to failure when the robotics approaches the terminal point.
- The current discretization time for system dynamics update is the same as the simulation timestep. Decreasing this value may also result in failure.
