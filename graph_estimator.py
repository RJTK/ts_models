'''
Module used for estimating the underlying granger causality graph.
'''

import numpy as np

from cross_validation import cx_validate_opt
from data_manipulation import build_YZ, adj_matrix, split_data

F_TRAIN = 0.7 #% for training
F_TEST = 1.0 #F_TEST - F_TRAIN = % for testing
F_VERIF = 1.0 #This should always be 1

#Parameters for cv optimizer
LAMBDA_MIN = 0.0001
LAMBDA_MAX = 5000

def estimate_gcg(D, model, p, T, delta = 0, ret_cv_result = False,
                 ret_benchmark_err = False, **kwargs):
  '''
  Estimates a GCG by thresholding an autoregressive model.
  '''
  D_train, I_train, D_test, I_test, D_verif, I_verif = split_data(
  D, T, F_TRAIN, F_TEST, F_VERIF)
  Y_train, Z_train = build_YZ(D_train, I_train, p)
  Y_test, Z_test = build_YZ(D_test, I_test, p)
  N_test = Y_test.size

  B_hat, lmbda_star, err_star = cx_validate_opt(Y_train, Z_train,
                                                Y_test, Z_test,
                                                model,
                                                lmbda_min = LAMBDA_MIN,
                                                lmbda_max = LAMBDA_MAX,
                                                **kwargs)
  A_hat = adj_matrix(B_hat, p, delta)

  ret = []
  if ret_cv_result:
    ret.extend([A_hat, lmbda_star, err_star / N_test])
  if ret_benchmark_err:
    err_mean = np.linalg.norm(Y_test, ord = 'fro')**2 / N_test
    ret.append(err_mean)

  if ret_cv_result or ret_benchmark_err:
    return tuple(ret)
  else:
    return A_hat

def edge_density(A):
  '''
  Simply calculates the proportion of edges present in A, not
  counting self-loops.  That is, pe = sum(A) / (n^2 - n)
  '''
  n = A.shape[0]
  assert n == A.shape[1] #It must be square
  pe = float(np.sum(A) - np.sum(A[np.diag_indices(n)])) / (n**2 - n)
  assert pe >= 0 and pe <= 1 #It's a proportion
  return pe
