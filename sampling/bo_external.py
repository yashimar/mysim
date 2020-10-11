import GPyOpt
import numpy as np
from numpy.random import seed
from matplotlib import pyplot as plt
from core_tool import *

def Help():
  pass

def Run(ct,*args):
  func = GPyOpt.objective_examples.experiments2d.branin()
  domain = [{"name": "var1", "type": "continuous", "domain": func.bounds[0]}, 
            {"name": "var2", "type": "continuous", "domain": func.bounds[1]}]

  X_init = np.array([[0.0,0.0],[0.5,0.0],[1.0,0.0]])
  Y_init = func.f(X_init)

  iter_count = 20
  current_iter = 0
  X_step = X_init
  Y_step = Y_init

  while current_iter < iter_count:
    bo_step = GPyOpt.methods.BayesianOptimization(
      f = None,
      domain = domain, 
      X = X_step, 
      Y = Y_step
    )
    x_next = bo_step.suggest_next_locations()
    y_next = func.f(x_next)

    X_step = np.vstack((X_step, x_next))
    Y_step = np.vstack((Y_step, y_next))

    current_iter += 1

  # bo_step.plot_acquisition()
  print(X_step)
  print(Y_step)

  # x = np.arange(0.0, 1.0, 0.01)
  # y = func.f(x)

  # plt.figure()
  # plt.plot(x, y)
  # for i, (xs, ys) in enumerate(zip(X_step, Y_step)):
  #   plt.plot(xs, ys, 'rD', markersize=10 + 20 * (i+1)/len(X_step))