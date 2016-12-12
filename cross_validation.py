import sys

import numpy as np
from scipy import optimize as sp_opt
from progressbar import Bar, Percentage, ETA, ProgressBar, SimpleProgress

CX_VAL_TOL = 1e-4

def cx_validate_opt(Y_train, Z_train, Y_test, Z_test, f, lmbda_min = 0,
                    lmbda_max = 100, **kwargs):
  '''
  Train the model given by f on the data (Y_train, Z_train) and uses
  scipy.optimize to search for the set of hyper parameters that
  provide the best performance on the test data (Y_test, Z_test).

  Currently set up to just minimize over a single hyperparameter.
  '''
  print 'Crossvalidating %s' % f.__name__
  def opt_f(lmbda):
    B = f(Y_train, Z_train, lmbda, **kwargs)
    Y_hat_test = np.dot(B, Z_test)
    err_test = np.linalg.norm(Y_test - Y_hat_test, 'fro')**2
    print 'lmbda = %f\r' % lmbda,
    sys.stdout.flush()
    return err_test
  
  #Replace this with optimize.minimize for a vector of params
  result = sp_opt.minimize_scalar(opt_f,
                                  bounds = (lmbda_min, lmbda_max),
                                  method = 'brent',
                                  tol = CX_VAL_TOL)
#                                  options = {'xatol': CX_VAL_TOL})
  print lmbda_min
  lmbda_star = result.x
  print 'lmbda_star = %f\n' % lmbda_star

  B_star = f(Y_train, Z_train, lmbda_star, **kwargs)
  Y_hat_star = np.dot(B_star, Z_test)
  err_star = np.linalg.norm(Y_test - Y_hat_star, 'fro')**2

  return B_star, lmbda_star, err_star
  
def cx_validate(Y_train, Z_train, Y_test, Z_test, Lmbda, f, 
                ret_path = False, **kwargs):
  '''
  Train the model given by f on the data (Y_train, Z_train) and then
  cycles through the set of parameters given in Lmbda to find the best
  performing set of hyperparameters on the test data (Y_test, Z_test).
  '''
  errs = []
  widgets = [Percentage(), ' ', Bar(), ' ', ETA()]
  pbar = ProgressBar(widgets = widgets, maxval = len(Lmbda))
  pbar.start()
  min_err = np.inf

  if ret_path:
    B_path = []
  for lmbda_i, lmbda in enumerate(Lmbda):
    pbar.update(lmbda_i)
    B = f(Y_train, Z_train, lmbda, **kwargs)
    if ret_path:
      B_path.append(B)

    Y_hat_test = np.dot(B, Z_test)
    err_test = np.linalg.norm(Y_test - Y_hat_test, 'fro')**2
    errs.append(err_test)
    try:
      if(err_test < min_err):
        B_star = B
        min_err = err_test
        lmbda_star = lmbda
    except NameError:
      B_star = B
      min_err = err_test
      lmbda_star = lmbda
  pbar.finish()

  if ret_path:
    B_path = np.dstack(B_path)
    return B_star, lmbda_star, errs, B_path

  return B_star, lmbda_star, errs
