# parameter_scheduling_mpc_ros
ROS implementation of the MPC controller with scheduled parameters for differential-drive wheeled mobile robot

## Dependencies
- CasADi: https://web.casadi.org/

## MDP data generation
MDP data can be generated by our parameter scheduling MPC script: https://github.com/MLCS-Yonsei/parameter_scheduling_mpc.git

## Launch
```
roslaunch parameter_scheduling_mpc parameter_scheduling_mpc.launch
```

## Optional arguments

- loop_rate: default="20"
- use_odom: default="false"
- regularization_u: default="0.001"
- regularization_v: default="0"
- regularization_omega: default="0"
- step_time: default="0.05"
- time_horizon: default="10"
- max_wheel_angular_vel: default="2.0"
- max_wheel_angular_acc: default="1.0"
- max_linear_vel: default="0.2"
- max_angular_vel: default="0.49"
- wheel_radius: default="0.1"
- wheel_separation: default="0.653"
- mdp_data: default="$(find parameter_scheduling_mpc)/bin/mdp_cartesian_10"
- RK4: default="false"
- cmd_vel_topic: default="/cmd_vel"
- base_frame: default="/base_link"
- xy_tol: default="0.05"
- yaw_tol: default="0.01"
- cost_r: default="1.0"
- cost_alpha: default="1.0"
- cost_phi: default="1.0"
- warm_start: default="false"
- mode: default="cartesian"
- minimize_rot_err: default="true"
- publish_path: default="false"
- switching: default="false"
