'''
Module used for estimating the underlying granger causality graph.
'''

import numpy as np

from cross_validation import cx_validate_opt
from data_manipulation import build_YZ, adj_matrix, split_data

#Default values for training and testing
F_TRAIN = 0.7 #% for training
F_TEST = 1.0 #F_TEST - F_TRAIN = % for testing
F_VERIF = 1.0 #This should always be 1

#Parameters for cv optimizer
LAMBDA_MIN = 0.0001
LAMBDA_MAX = 5000

def vote_AoN(A):
  '''
  Produces the final A_hat estimate by an All or Nothing approach.
  That is, we take A_hat = A[0] & A[1] & ... & A[-1].  Each A[i]
  should be a boolean matrix of the same size n by n.
  '''
  #equivalent to:
  #init A_hat
  #for Ai in A
  #  A_hat = np.logical_and(A_hat, Ai)
  A_hat = np.logical_and.reduce(A)
  return A_hat

def estimate_gcg_vote(D, model, p, T, vote_func, K, F_train = 0.7,
                      delta = 0, ret_cv_result = False,
                      ret_benchmark_err = False, **kwargs):
  '''
  Breaks the entire training set up into K portions.  We then fit
  the model, and apply a threshold with parameter delta to the sum of
  the B_hat matrices.  The resulting A_hat_k sequences is later passed
  as a list to the vote_func to decide the final A_hat.  Each subset
  of the data is broken into proportion F_train for training and
  1 - F_train for testing and cross validation.
  '''
  all_A_hat = []
  Tk = int(T) / int(K)
  #Note that some of the data is discarded if the integers don't divide nicely
  lmbda_star = 0
  rel_err_star = 0
  rel_err_mean = 0
  for k in range(K):
    D_train, I_train, D_test, I_test, D_verif, I_verif = split_data(
      D.iloc[k*Tk : (k + 1)*Tk], Tk, F_train = F_train, F_test = 1.0,
      F_verif = 1.0
    )
    Y_train, Z_train = build_YZ(D_train, I_train, p)
    Y_test, Z_test = build_YZ(D_test, I_test, p)
    N_test = Y_test.size

    rel_err_mean_k = np.linalg.norm(Y_test, ord = 'fro')**2 / N_test
    B_hat, lmbda_star_k, err_star_k = cx_validate_opt(Y_train, Z_train,
                                                      Y_test, Z_test,
                                                      model,
                                                      lmbda_min = LAMBDA_MIN,
                                                      lmbda_max = LAMBDA_MAX,
                                                      **kwargs)
    A_hat_k = adj_matrix(B_hat, p, delta)
    rel_err_mean += rel_err_mean_k
    all_A_hat.append(A_hat_k)
    lmbda_star += lmbda_star_k
    rel_err_star += err_star_k / N_test

  #Apply vote function
  A_hat = vote_func(all_A_hat)

  lmbda_star /= K
  rel_err_star /= K
  rel_err_mean /= K

  ret = []
  if ret_cv_result:
    ret.extend([A_hat, lmbda_star, rel_err_star])
  if ret_benchmark_err:
    ret.append(rel_err_mean)

  if ret_cv_result or ret_benchmark_err:
    return tuple(ret)
  else:
    return A_hat

def estimate_gcg(D, model, p, T, delta = 0, ret_cv_result = False,
                 ret_benchmark_err = False, f_start = None, f_train = None,
                 **kwargs):
  '''
  Estimates a GCG by thresholding an autoregressive model.
  '''
  if f_start is None:
    f_start = 0.0
  if f_train is None:
    f_train = F_TRAIN

  D_train, I_train, D_test, I_test, D_verif, I_verif = split_data(
    D, T, F_train = f_train, F_test = 1.0, F_verif = 1.0, F_start = f_start)
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
  Simply calculates the proportion of edges present in A.
  '''
  n = A.shape[0]
  assert n == A.shape[1] #It must be square
  pe = float(np.sum(A)) / n**2
#  pe = float(np.sum(A) - np.sum(A[np.diag_indices(n)])) / (n**2 - n)
  assert pe >= 0 and pe <= 1 #It's a proportion
  return pe

def false_discovery_rate(A_hat, A):
  '''sum(False Positive) / max(1, sum(Test Outcome Positive))'''
  fp = float(np.sum(np.logical_and(A_hat, A == 0)))
  num_pos_hat = np.sum(A_hat)
  fdr = fp/num_pos_hat
  return fdr

def false_positive_rate(A_hat, A):
  num_neg = np.sum(A == 0)
  fpr = np.sum(np.logical_and(A_hat, A == 0)) / float(num_neg)
  return fpr

def false_negative_rate(A_hat, A):
  fn = np.sum(np.logical_and(A_hat == 0, A))
  num_pos = np.sum(A)
  fnr = fn / num_pos
  return fnr

def true_positive_rate(A_hat, A): #also called "recall"
  tp = float(np.sum(np.logical_and(A_hat, A)))
  num_pos = np.sum(A)
  tpr = tp / num_pos
  return tpr

def true_negative_rate(A_hat, A):
  tn = float(np.sum(np.logical_and(A_hat == 0, A == 0)))
  num_neg = np.sum(A == 0)
  tnr = tn / num_neg
  return tnr

def precision(A_hat, A): #also called 'Positive Predictive Value' (PPV)
  tp = float(np.sum(np.logical_and(A_hat, A)))
  num_pos_hat = np.sum(A_hat)
  prec = tp / num_pos_hat
  return prec

def F1_score(A_hat, A):
  tpr = true_positive_rate(A_hat, A)
  prec = precision(A_hat, A)
  f1 = 2*(tpr * prec) / (tpr + prec)
  return f1
