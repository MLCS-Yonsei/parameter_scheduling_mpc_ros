import time
import numpy as np
from casadi import *
from variables import Variables, EPS
from utils import normalize_angle, RK2, RK4

__all__ = ['NmpcPolar']



class NmpcPolar:
    
  def __init__(
    self, T=0.05, N=20,
    R_u=0.1*np.eye(2), R_vel=0.1*np.eye(2),
    w_max=2, a_max=4, v_max=None, omega_max=None, vel_constraint='quadratic',
    max_iter=100,
    ode='RK4',
    wheel_rad=0.1,
    wheel_separation=0.653
  ):

    self.T = T # Time horizon
    self.N = N # number of control intervals
    self.max_iter = max_iter

    self.wheel_rad = wheel_rad
    self.wheel_separation = wheel_separation
    self.w_max = w_max
    if a_max is None or self.w_max is None:
      self.dw_max = None
    else:
      self.dw_max = a_max * T
    if v_max is None:
      self.v_max = self.wheel_rad * self.w_max
      self.v_constraint = False
    else:
      self.v_max = v_max
      self.v_constraint = True
    if omega_max is None:
      self.omega_max = self.wheel_rad * self.w_max * 2 / self.wheel_separation
      self.omega_constraint = False
    else:
      self.omega_max = omega_max
      self.omega_constraint = True
    if vel_constraint is 'quadratic':
      self.vel_constraint = False
    else:
      self.vel_constraint = True
      self.v_constraint = False
      self.omega_constraint = False
    self.dv_max = self.wheel_rad * self.dw_max
    self.domega_max = self.wheel_rad * self.dw_max * 2 / self.wheel_separation

    X_max = np.array([np.inf, np.inf, np.inf])
    X_min = np.array([0.0, -np.inf, -np.inf])
    U_max = np.array([self.w_max, self.w_max])
    U_min = -U_max
    dU_max = np.array([self.dw_max, self.dw_max])
    dU_min = -dU_max

    #setting up symbolic variables for states 
    r = SX.sym('r')
    alpha = SX.sym('alpha')
    phi = SX.sym('phi')
    states = vertcat(r, alpha, phi)
    self.n_states = states.size1()
    
    #setting up symbolic variables for control inputs 
    omega_l = SX.sym('omega_l')
    omega_r = SX.sym('omega_r')
    controls = vertcat(omega_l, omega_r)
    self.n_controls = controls.size1()

    v = SX.sym('v')
    omega = SX.sym('omega')
    H = self.wheel_rad * vertcat(
      horzcat(1, 1) / 2,
      horzcat(-1, 1) / self.wheel_separation
    )
    rhs = vertcat(-v*cos(alpha), v*sin(alpha)/(r+EPS)-omega, -v*sin(alpha)/(r+EPS)) # system r.h.s

    f = Function('f', [states, v, omega], [rhs]) # nonlinear mapping function f(x,u), system equation 
    U = Variables('U') # Decision variables (controls), optimal control input as determined by optimizer 
    P = SX.sym('P', self.n_states + self.n_controls + 4) # parameters (which include the initial and the reference/final desired state of the robot, and the cost function parameters)
    
    X = Variables('X') # State variables

    obj = 0 # Objective function
    g = Variables('g') # constraints
    if np.trace(np.dot(R_u.T, R_u)) == 0:
      R_u = None
    else:
      R_u = horzcat(R_u[0, :], R_u[1, :])
    if np.trace(np.dot(R_vel.T, R_vel)) == 0:
      R_vel = None
    else:
      R_vel = horzcat(R_vel[0, :], R_vel[1, :])

    st = SX.sym('X0', self.n_states)
    X.add_inequality(st, ub=X_max, lb=X_min) # initial state
    g.add_equality(st-P[0:3]) # initial condition constraints
    # compute objective symbollically this is what the solver will minimize 
    for k in range(0, self.N):
      con = SX.sym('U'+str(k), self.n_controls)
      U.add_inequality(con, ub=U_max, lb=U_min)
      if self.dw_max < np.inf:
        if k == 0:
          dcon = P[5] * (con - P[3:5])
        else:
          dcon = con - con_prev
        g.add_inequality(dcon, ub=dU_max, lb=dU_min)
      vel = mtimes(H, con)
      if self.v_constraint:
        g.add_inequality(
          vel[0],
          ub=self.v_max * np.ones((1,)),
          lb=-self.v_max * np.ones((1,)),
        )
      if self.omega_constraint:
        g.add_inequality(
          vel[1],
          ub=self.omega_max * np.ones((1,)),
          lb=-self.omega_max * np.ones((1,)),
        )
      if self.vel_constraint:
        g.add_inequality(
          (vel[0]/self.v_max) ** 2 + (vel[1]/self.omega_max) ** 2,
          ub=np.ones((1,)), lb=None
        )
      obj += P[6] * st[0] ** 2
      obj += P[7] * (1 - cos(st[1]))
      obj += P[8] * (1 - cos(st[2]))
      if R_u is not None:
        obj += mtimes(con.T,mtimes(R_u, con))
      if R_vel is not None:
        obj += mtimes(vel.T,mtimes(R_vel,vel))
      st_next = SX.sym('X'+str(k+1), self.n_states)
      X.add_inequality(st_next, ub=X_max, lb=X_min)
      if ode=='RK4':
        st_prediction = RK4(f, T, st, vel[0], vel[1])
      if ode=='RK2':
        st_prediction = RK2(f, T, st, vel[0], vel[1])
      else:
        st_prediction = st + f(st, vel[0], vel[1]) * T
      g.add_equality(st_next - st_prediction) # compute kinematic constraints
      st = st_next
      con_prev = con
        
    OPT_variables = X + U

    opts = {
      'ipopt.max_iter': self.max_iter,
      'ipopt.print_level': 0,
      'print_time': 0,
      'ipopt.acceptable_tol': 1e-8,
      'ipopt.acceptable_obj_change_tol': 1e-6
    }

    nlp_prob = {'f':obj, 'x':OPT_variables.var, 'g':g.var, 'p':P}
    self.solver = nlpsol('solver', 'ipopt', nlp_prob, opts)

    self.args = {'lbg':g.lb,'ubg':g.ub,'lbx':OPT_variables.lb,'ubx':OPT_variables.ub} 
    self.u0 = np.zeros((N, self.n_controls))

  @staticmethod
  def __dx(x, v, omega):
    return np.array(
      [-v*np.cos(x[1]), v*np.sin(x[1])/(x[0]+EPS)-omega, -v*np.sin(x[1])/(x[0]+EPS)]
    )


  def solve(self, state, theta, u_prev=None, return_trajectory=False, warm_start=0):
    x_init = [
      [state[0] - EPS if state[0] > EPS else 0.],
      [state[1]],
      [state[2] - state[1]]
      #[state[2]]
    ]

    if u_prev is None:
      self.args['p'] = vertcat(
        np.array(x_init),
        np.zeros((2,1)),
        0,
        np.reshape(theta, (3,1))
      )
    else:
      self.args['p'] = vertcat(
        np.array(x_init),
        np.reshape(u_prev, (2,1)),
        1,
        np.reshape(theta, (3,1))
      )
    if warm_start==1:
      u_init = []
      x_warm = np.reshape(x_init, (-1))
      u_warm = u_prev
      for k in range(0, self.N):
        if x_warm[0] < EPS:
          if x_warm[0] < 0:
            x_warm[0] = 0
          v = 0
          if x_warm[1] + x_warm[2] == 0.:
            omega = 0
          else:
            omega = -self.omega_max * (x_warm[1] + x_warm[2]) / np.pi
        else:
          cr = np.cos(x_warm[1]) * (x_warm[0] + EPS)
          sr = np.sin(x_warm[1]) * (x_warm[0] + EPS)
          if cr >=0:
            omega = self.omega_max * x_warm[1] * 2 / np.pi
          else:
            omega = -self.omega_max * x_warm[1] * 2 / np.pi
          v = self.v_max * min(cr, 1 - np.abs(omega / self.omega_max))
        if self.vel_constraint:
          scale = max(np.sqrt((v / self.v_max)**2 + (omega / self.omega_max)**2), 1)
          v /= scale
          omega /= scale
        else:
          scale = max(np.abs(v / self.v_max) + np.abs(omega / self.omega_max), 1)
          v /= scale
          omega /= scale
        if self.dw_max < np.inf:
          if u_warm is None:
            u_warm = np.array(
              [
                (v-omega*self.wheel_separation/2)/self.wheel_rad,
                (v+omega*self.wheel_separation/2)/self.wheel_rad
              ]
            )
            u_warm[0] = np.sign(u_warm[0]) * min(np.abs(u_warm[0]), self.w_max)
            u_warm[1] = np.sign(u_warm[1]) * min(np.abs(u_warm[1]), self.w_max)
          else:
            du = np.array(
              [
                (v-omega*self.wheel_separation/2)/self.wheel_rad,
                (v+omega*self.wheel_separation/2)/self.wheel_rad
              ]
            ) - u_warm
            du[0] = np.sign(du[0]) * min(np.abs(du[0]), self.dw_max)
            du[1] = np.sign(du[1]) * min(np.abs(du[1]), self.dw_max)
            u_warm += du
            u_warm[0] = np.sign(u_warm[0]) * min(np.abs(u_warm[0]), self.w_max)
            u_warm[1] = np.sign(u_warm[1]) * min(np.abs(u_warm[1]), self.w_max)
          v = self.wheel_rad * (u_warm[0] + u_warm[1]) / 2
          omega = self.wheel_rad * (u_warm[1] - u_warm[0]) / self.wheel_separation
        else:
          u_warm = np.array(
            [
              (v-omega*self.wheel_separation/2)/self.wheel_rad,
              (v+omega*self.wheel_separation/2)/self.wheel_rad
            ]
          )
        x_warm = RK4(self.__dx, self.T, x_warm, v, omega)
        x_warm[2] = normalize_angle(x_warm[2])
        x_init.append([x_warm[0]])
        x_init.append([x_warm[1]])
        x_init.append([x_warm[2]])
        u_init.append([u_warm[0]])
        u_init.append([u_warm[1]])
      self.args['x0'] = vertcat(
        np.array(x_init),
        np.array(u_init)
      ) # initial value of the optimization variables, w0 is the optimization variable
    elif warm_start==2:
      u_init = []
      x_warm = np.reshape(x_init, (-1))
      u_warm = u_prev
      for k in range(0, self.N):
        calpha = np.cos(x_warm[1])
        salpha = np.sin(x_warm[1])
        v = theta[0] * (x_warm[0] + EPS) * calpha
        if np.abs(x_warm[1]) < EPS:
          omega = x_warm[1] - theta[2] * x_warm[2] / theta[1]
        else:
          omega = calpha * salpha - theta[2] * calpha * salpha * x_warm[2] / theta[1] / x_warm [1] + x_warm[1]
        if self.vel_constraint:
          scale = max(np.sqrt((v / self.v_max)**2 + (omega / self.omega_max)**2), 1)
          v /= scale
          omega /= scale
        else:
          scale = max(np.abs(v / self.v_max) + np.abs(omega / self.omega_max), 1)
          v /= scale
          omega /= scale
        if self.dw_max < np.inf:
          if u_warm is None:
            u_warm = np.array(
              [
                (v-omega*self.wheel_separation/2)/self.wheel_rad,
                (v+omega*self.wheel_separation/2)/self.wheel_rad
              ]
            )
            u_warm[0] = np.sign(u_warm[0]) * min(np.abs(u_warm[0]), self.w_max)
            u_warm[1] = np.sign(u_warm[1]) * min(np.abs(u_warm[1]), self.w_max)
          else:
            du = np.array(
              [
                (v-omega*self.wheel_separation/2)/self.wheel_rad,
                (v+omega*self.wheel_separation/2)/self.wheel_rad
              ]
            ) - u_warm
            du[0] = np.sign(du[0]) * min(np.abs(du[0]), self.dw_max)
            du[1] = np.sign(du[1]) * min(np.abs(du[1]), self.dw_max)
            u_warm += du
            u_warm[0] = np.sign(u_warm[0]) * min(np.abs(u_warm[0]), self.w_max)
            u_warm[1] = np.sign(u_warm[1]) * min(np.abs(u_warm[1]), self.w_max)
          v = self.wheel_rad * (u_warm[0] + u_warm[1]) / 2
          omega = self.wheel_rad * (u_warm[1] - u_warm[0]) / self.wheel_separation
        else:
          u_warm = np.array(
            [
              (v-omega*self.wheel_separation/2)/self.wheel_rad,
              (v+omega*self.wheel_separation/2)/self.wheel_rad
            ]
          )
        x_warm = RK4(self.__dx, self.T, x_warm, v, omega)
        x_warm[2] = normalize_angle(x_warm[2])
        x_init.append([x_warm[0]])
        x_init.append([x_warm[1]])
        x_init.append([x_warm[2]])
        u_init.append([u_warm[0]])
        u_init.append([u_warm[1]])
      self.args['x0'] = vertcat(
        np.array(x_init),
        np.array(u_init)
      ) # initial value of the optimization variables, w0 is the optimization variable
    else:
      X0 = repmat(np.array(x_init), 1, self.N+1)
      self.args['x0'] = vertcat(
        reshape(X0.T, self.n_states*(self.N+1), 1),
        reshape(self.u0.T, self.n_controls*self.N, 1)
      ) # initial value of the optimization variables, w0 is the optimization variable

    sol = self.solver(
      x0  = self.args['x0'],
      lbx = self.args['lbx'],
      ubx = self.args['ubx'],
      lbg = self.args['lbg'],
      ubg = self.args['ubg'],
      p   = self.args['p']
    )
    U = np.reshape(sol['x'][self.n_states*(self.N+1):], (self.N, self.n_controls)).astype(np.float32)
    if return_trajectory:
      states = []
      X = np.reshape(sol['x'][:self.n_states*(self.N+1)], (self.N + 1, self.n_states))
      for x in X:
        states.append([
          x[0],
          x[1],
          x[1] + x[2]
          #x[2]
        ])
      return np.array(states, dtype=np.float32), U
    else:
      return U[0, :]



if __name__ == '__main__':

  import numpy as np

  tic = time.time()
  nmpc = NmpcPolar(
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
