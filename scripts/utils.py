from sys import stdout
from numpy import allclose, sin, cos, arctan2

__all__ = ['normalize_angle', 'RK2', 'RK4', 'Verbose', 'ArrEq']



def normalize_angle(angle):

  return arctan2(sin(angle), cos(angle))


def RK2(function, duration, state, *args):

  k1 = duration * function(state, *args)
  k2 = duration * function(state + k1, *args)
  sol = state + (k1 + k2) / 2

  return sol


def RK4(function, duration, state, *args):

  k1 = duration * function(state, *args)
  k2 = duration * function(state + k1 / 2, *args)
  k3 = duration * function(state + k2 / 2, *args)
  k4 = duration * function(state + k3, *args)
  sol = state + (k1 + 2 * k2 + 2 * k3 + k4) / 6

  return sol


class Verbose:

  def __init__(self, *args):
    self.verbose = args[0]
    self.lenStr = 0

  def __call__(self, string):
    if self.verbose:
      stdout.write('\r'+' '*self.lenStr+'\r')
      stdout.flush()
      self.lenStr = 0 if string[-2:]=='\n' else len(string)
      stdout.write(string)
      stdout.flush()



class ArrEq:

  def __init__(self, obj):
    self.obj = obj

  def __eq__(self, other):
    if self.obj.shape == other.shape:
      return allclose(self.obj, other)
    else:
      return False

