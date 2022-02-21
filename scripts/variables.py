import numpy as np
from casadi import *

__all__ = ['Variables']



EPS = np.finfo(float).eps


class Variables:

  '''
  var: Symbolic variables
  ub: Upper bounds for symbolic variables
  lb: Lower bounds for symbolic variables
  add_inequality(constraints, ub, lb): Adds symbolic variables and its bounds.
  add_equality(constraints): Adds symbolic variables. Bounds of the variables are set to be 0
  '''

  def __init__(self, name=None, var=None, ub=None, lb=None):

    self.name = name

    if type(var)==type(None):
      self.var = SX.zeros(0)
      self.ub = np.array([])
      self.lb = np.array([])
    else:
      self.var = var
      if type(ub)==type(None):
        self.ub = float('inf')*np.ones(var.shape)
      else:
        self.ub = ub
      if type(lb)==type(None):
        self.lb = -float('inf')*np.ones(var.shape)
      else:
        self.lb = lb


  def __add__(self, other):

    temp = Variables(self.name + '+' + other.name)
    temp.var = vertcat(self.var, other.var)
    temp.ub = np.concatenate([self.ub, other.ub])
    temp.lb = np.concatenate([self.lb, other.lb])

    return temp


  def __getitem__(self, key):

    return {
      'param':self.var[key],
      'ub':self.ub[key],
      'lb':self.lb[key]
    }


  def add_inequality(self, constraints, ub=None, lb=None):

    self.var = vertcat(self.var, constraints)
    if type(ub)==type(None):
      self.ub = np.concatenate([self.ub, float('inf')*np.ones([constraints.shape[0]])])
    else:
      self.ub = np.concatenate([self.ub, ub])
    if type(lb)==type(None):
      self.lb = np.concatenate([self.lb, -float('inf')*np.ones([constraints.shape[0]])])
    else:
      self.lb = np.concatenate([self.lb, lb])


  def add_equality(self, constraints):

    self.var = vertcat(self.var, constraints)
    self.ub = np.concatenate([self.ub, np.zeros([constraints.shape[0]])])
    self.lb = np.concatenate([self.lb, np.zeros([constraints.shape[0]])])
