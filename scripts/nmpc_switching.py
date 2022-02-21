import time
import numpy as np
from casadi import *
from nmpc_cartesian import NmpcCartesian
from nmpc_polar import NmpcPolar
from nmpc_rotation import NmpcRotation
from utils import normalize_angle, RK2, RK4

__all__ = ['NmpcPolar']



class NmpcSwitching:
    
  def __init__(
    self, T=0.05, N=20,
    R_u=0.1*np.eye(2), R_vel=0.1*np.eye(2),
    w_max=2, a_max=4, v_max=None, omega_max=None, vel_constraint='quadratic',
    max_iter=100,
    ode='RK4',
    r_tol=5e-3,
    mode='polar',
    wheel_rad=0.1,
    wheel_separation=0.653
  ):
    self.nmpc_translation = NmpcPolar(
      T=T, N=N, R_u=R_u, R_vel=R_vel, max_iter=max_iter, ode=ode,
      w_max=w_max, a_max=a_max, v_max=v_max, omega_max=omega_max, vel_constraint=vel_constraint,
      wheel_rad=wheel_rad, wheel_separation=wheel_separation
    ) if mode=='polar' else NmpcCartesian(
      T=T, N=N, R_u=R_u, R_vel=R_vel, max_iter=max_iter, ode=ode,
      w_max=w_max, a_max=a_max, v_max=v_max, omega_max=omega_max, vel_constraint=vel_constraint,
      wheel_rad=wheel_rad, wheel_separation=wheel_separation
    )
    self.nmpc_rotation = NmpcRotation(
      T=T, N=N, R_u=R_u, R_vel=R_vel, max_iter=max_iter, ode=ode,
      w_max=w_max, a_max=a_max, v_max=v_max, omega_max=omega_max, vel_constraint=vel_constraint,
      wheel_rad=wheel_rad, wheel_separation=wheel_separation
    )
    self.r_tol = r_tol

  def solve(self, state, theta, u_prev=None, return_trajectory=False, warm_start=0):
    if state[0] < self.r_tol:
      sol = self.nmpc_rotation.solve(
        state, theta[1]+theta[2], u_prev=u_prev, return_trajectory=return_trajectory, warm_start=0
      )
    else:
      sol = self.nmpc_translation.solve(
        state, theta, u_prev=u_prev, return_trajectory=return_trajectory, warm_start=0
      )
    return sol


if __name__ == '__main__':

  import numpy as np

  tic = time.time()
  nmpc = NmpcSwitching(
    N=30,
    T=0.05,
    R_u=0*np.eye(2),
    R_vel=0*np.eye(2),
    w_max=4,
    a_max=4,
    v_max=0.2,
    omega_max=0.2 * 0.8 * 2 / 0.653,
    max_iter=100,
    ode='RK4'
  )
  print('setup time:', time.time()-tic, 'sec')
  tic = time.time()
  X, U = nmpc.solve(
    [0.0001, 0, np.pi/2], # initial state
    [1, 1, 1],             # cost function weight
    u_prev=[0, 0],
    warm_start=0,
    return_trajectory=True,
  )
  print('control input:', U[0, :])
  print('solution time:', time.time()-tic, 'sec')
  traj = []
  for x in X:
    traj.append([
      x[0] * np.cos(np.pi + x[1] - x[2]),
      x[0] * np.sin(np.pi + x[1] - x[2]),
      -x[2]
    ])
  traj = np.array(traj)
  print('predicted pose:')
  print(traj)
