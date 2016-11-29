import numpy as np
import spams as spm

from spams import lasso, fistaFlat, proximalFlat
from scipy.optimize import minimize, differential_evolution
from numpy.linalg import lstsq
from scipy.linalg import lu_factor, lu_solve

from data_manipulation import *

SPAMS_TRACE_TOL = 5e-4

#-------TIME SERIES MODELS Y_t^ = sum_i B_i * Y_(t - i)----------
#Possible Additions:
#--Dantzig Selector (for n >> T)
#--Depth-Wise GLASSO
#--Time Varying LASSO
#--SLOPE (Candes)
def fit_ols(Y, Z):
  '''
  B = argmin_B (0.5)||Y - BZ||_F^2
  '''
  B, R, rk, s = lstsq(Z.T, Y.T)
  return B.T

def fit_olst(Y, Z, lmbda, ZZT = None):
  '''
  B = argmin_B (0.5)||Y - BZ||_F^2 + lmbda||B||_F^2
  '''
  #It's obviously bad to directly form the below inverse
  #but since we have the lmbda*I term it shouldn't be
  #too poorly conditioned anyway.
  if ZZT is None:
    ZZT = np.dot(Z, Z.T)
  tmp = lmbda*np.eye(ZZT.shape[0]) + ZZT
  tmp = np.linalg.inv(tmp)
  YZT = np.dot(Y, Z.T)
  B = np.dot(YZT, tmp)
  return B

def spams_trace_setup(Y = None, Z = None):
  '''
  Function closure to keep the previous B0 as the starting point for
  cross validation
  '''
  def spams_trace(Y, Z, lmbda):
    '''
    B = argmin_B (0.5)||Y - BZ||_F^2 + lmbda||B||_*
    '''
    if spams_trace.B0 is not None:
      B0 = spams_trace.B0
    else:
      B0 = fit_olst(Y, Z, lmbda)
      B0 = np.asfortranarray(B0.T)
      spams_trace.B0 = B0

    Y_spm = np.asfortranarray(Y.T)
    Z_spm = np.asfortranarray(Z.T)
    B, results = fistaFlat(Y_spm, Z_spm, B0, True, loss = 'square',
                           regul = 'trace-norm', lambda1 = lmbda,
                           verbose = False, tol = SPAMS_TRACE_TOL)
    spams_trace.B0 = B
    return B.T
  if Y is not None and Z is not None:
    B0 = fit_olst(Y, Z, 1)
    B0 = np.asfortranarray(B0.T)
    spams_trace.B0 = B0
  else:
    spams_trace.B0 = None
  return spams_trace

def spams_glasso_setup(Y = None, Z = None):
  '''
  Function closure to keep the previous B0 as the starting point for
  cross validation
  '''
  def spams_glasso(Y, Z, lmbda):
    '''
    b = argmin_b (0.5)||Y - [Z^T (x) I]b||_2^2 + lmbda sum_g\in G ||b_g||_2
    (Z^T (x) I) = vec(BZ) (kron product) this is needed to specify groups.
    '''
    if spams_glasso.b0 is not None:
      b0 = spams_glasso.b0
    else:
      B0 = fit_olst(Y, Z, lmbda)
      b0 = vec(B0)
      b0 = np.asfortranarray(b0.reshape((b0.shape[0], 1)))
      spams_glasso.b0 = b0

    n = Y.shape[0]
    y = vec(Y)
    Z = bsr_matrix(np.kron(np.eye(n), Z.T)) #This will be huge...

    y_spm = np.asfortranarray(y.reshape((y.shape[0], 1)))
    Z_spm = np.asfortranarray(Z)
    
    b, results = fistaFlat(y_spm, Z_spm, b0, True, loss = 'square',
                           regul = 'l1', lambda1 = lmbda,
                           verbose = False)
    spams_glasso.b0 = b
    B = np.array(b.reshape((n, n)))
    return B

  if Y is not None and Z is not None:
    B0 = fit_olst(Y, Z, 1)
    b0 = vec(B0)
    b0 = np.asfortranarray(b0.reshape((b0.shape[0], 1)))
    spams_glasso.b0 = b0
  else:
    spams_glasso.B0 = None
  return spams_glasso

def spams_lasso(Y, Z, lmbda):
  '''
  B = argmin_B (0.5)||Y - BZ||_F^2 + lmbda||B||_1
  '''
  Y_spm = np.asfortranarray(Y.T)
  Z_spm = np.asfortranarray(Z.T)
  B = lasso(Y_spm, Z_spm, lambda1 = lmbda, lambda2 = 0,
            mode = spm.PENALTY)
  B = B.toarray() #Convert from sparse array
  return B.T

def dwglasso(Y, Z, lmbda, eps = 1e-12, mu = .1):
  '''
  DWGLASS by ADMM.  lmbda is the regularization term, mu is a
  parameter of the algorithm.  NOTE: The names of lmbda and mu are
  reversed in the proximal algorithms set of notes.
  '''
  #Consider Forming this whole thing in terms of the tilde matrices
  #so that I only need to run B_to_Bt once at the start and
  #Bt_to_B once at the end.  These methods take about half the time.

  #NOTE: It may be possible to get faster convergence using a 'smart'
  #sequence of lmbdas.  However, it would then no longer be possible
  #to cache the LU factorization.
  n = Y.shape[0]
  p = Z.shape[0] / n

  ZZT = np.dot(Z, Z.T)
  R = (ZZT + np.eye(n*p) / mu).T
  PLU = lu_factor(R, overwrite_a = True) #CARE, R IS OVERWRITTEN

  ZYT = np.dot(Z, Y.T)

  def proxf(A):
    '''
    Proximal operator for f(B) = ||Y - BZ||_F^2
    A = Bz^(k) - Bu^(k)
    Returns: Bx^(k + 1)
    Solves: (PLU)B^T = ZYT + A^T / mu for B
    '''
    return (lu_solve(PLU, ZYT + A.T / mu)).T
  
  def proxg(V):
    '''
    Proximal operator for g(B) = mu * lmbda * sum_ij[ ||Bij||_2 ]
    V = Bx^(k + 1) + Bu^(k)
    Returns: Bu^(k + 1)
    '''
    V_spm = np.asfortranarray(V.T)
    Bu = proximalFlat(V_spm, return_val_loss = False, lambda1 = mu*lmbda,
                      regul = 'l1l2')
    Bu = np.array(Bu.T)
    return Bu

  def rel_err(Bxk, Bzk):
    return (1./(n**2)) * np.linalg.norm(Bxk - Bzk, 'f')**2

  Bz = fit_olst(Y, Z, lmbda) #Initial guess for Bz
  Bx, Bu = np.zeros_like(Bz), np.zeros_like(Bz)

  #PERFORM ADMM
  rel_err_k = rel_err(Bx, Bz)
  while rel_err_k > eps:
#    print '(1/n)||Bz - Bx||_F^2 = %f\r' % rel_err_k,
    A = Bz - Bu
    Bx = proxf(A)

    U = Bx + Bu
    Ut = B_to_Bt(U, n, p)
    Bzt = proxg(Ut)
    Bz = Bt_to_B(Bzt, n, p)

    Bu = Bu + Bx - Bz
    rel_err_k = rel_err(Bx, Bz)
#  print ''
  return Bx
#  return (Bx + Bz) / 2. #Could averaging remove zeros?

def dwglasso_nm(Y, Z, lmbda):
  '''Function I played with'''
  n = Y.shape[0]
  p = Z.shape[0] / n

  def cost(Y, Z, b, lmbda):
    '''Cost function'''
    B = b.reshape((n, n*p)) #Un-vectorize
    Bt = B_to_Bt(B, n, p) #speeds up calculating R

    Y_hat = np.dot(B, Z) #Current prediction
    L = np.linalg.norm(Y - Y_hat, 'fro')**2 #Loss
    R = np.sum(np.linalg.norm(Bt, axis = 0)) #Regularization
    return L + lmbda*R

  B0 = fit_olst(Y, Z, lmbda) #Initial guess
  b0 = vec(B0) #vectorize

#  res = minimize(lambda b : cost(Y, Z, b, lmbda), b0,
#                 method = 'SLSQP', tol = 1e-6, options = {'disp':True})
  res = differential_evolution(lambda b : cost(Y, Z, b, lmbda), 
                               bounds = [(-1.5, 1.5) for i in range(len(b0))],
                               polish = True, disp = True, tol = 1e-4)
  b = res.x
  B = b.reshape((n, n*p))
  return B
