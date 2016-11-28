import pickle

import numpy as np

#-------DATA MANIPULATION ROUTINES--------
def load_data(file_name):
  '''
  Loads a previously pickled set of data.  The loaded object should
  be a dictionary containing the parameters, the underlying causality
  graph, and the actual data.  This is designed to load in data
  generated by the data_synthesis routines.  See that file for details.
  '''
  f = open(file_name, 'rb')
  A = pickle.load(f)
  f.close()
  return A

def B_to_Bt(B, n, p):
  '''B = [B^(1) ... B^(p)], B^(i) \in R^(n, n)'''
  k = np.array(range(p))
  Bt = [B[i, j + n * k].reshape((p, 1))
        for i in range(n) for j in range(n)]
  Bt = np.hstack(Bt)
  return Bt

def Bt_to_B(Bt, n, p):
  '''Bt = [B_11 B_12 ... B_nn], B_ij \in R^p'''
  k = np.array(range(p))
  B = np.zeros([n, n*p])
  for i in range(n):
    for j in range(n):
      B[i, j + n*k] = Bt[:, n*i + j]
  return B

def Z_to_Zt(Z, p):
  '''
  Converts Z to Z_tilde
  '''
  n = Z.shape[0] / p
  k = np.array(range(p)) #0, n, ..., (p - 1)n
  Zt = [np.vstack([z[j + n * k] for z in Z.T]).T for j in range(n)]
  return Zt

def Zt_to_Z(Zt, p):
  '''
  Converts Zt back to Z
  '''
  #Pain in the ass, not sure I need it
  raise NotImplementedError
  return

def Y_to_Yt(Y):
  '''
  Converts Y to Yt.  Yt is just a list of the rows of Y
  '''
  Yt = [y for y in Y]
  return Yt

def Yt_to_Y(Yt):
  '''
  Converts Yt back to Y
  '''
  Y = np.vstack(Yt)
  return Y

def build_YZ(D, I, p):
  '''
  Builds the Y (output) and Z (input) matrices for the model from
  a pandas dataframe D and set of indices I.  We need to also provide
  the lag length of the model, p.

  Y_hat = BZ

  D: Pandas dataframe of data
  I: Set of indices to use
  p: Model lag length
  '''
  T = len(I)
  if T == 0:
    return np.array([]), np.array([])
  Y = np.array(D[p:]).T

  Z = np.array(D.ix[I[p - 1: : -1]]).flatten()
  for k in range(1, T - p):
    Zk = np.array(D.ix[I[k + p - 1: k - 1: -1]]).flatten()
    Z = np.vstack((Z, Zk))
  Z = Z.T
  return Y, Z

def split_data(D, T, F_train, F_test, F_verif):
  '''Splits the data into separate train, test and verify pieces'''
  I_train = D.index[0:int(T*F_train)]
  I_test = D.index[int(T*F_train):int(T*F_test)]
  I_verif = D.index[int(T*F_test):int(T*F_verif)]

  D_train = D.ix[I_train].copy()
  D_test = D.ix[I_test].copy()
  D_verif = D.ix[I_verif].copy()

  return D_train, I_train, D_test, I_test, D_verif, I_verif

def vec(A):
  '''Vectorizes A by stacking the columns.  A should be 2D'''
  return A.T.reshape(A.shape[0]*A.shape[1])

def adj_matrix(B, p):
  '''Obtain the adjacency matrix of the model B'''
  n = B.shape[0]
  A = sum(np.abs(B[0 : n, k*n : (k + 1)*n])
                for k in range(0, p))
  A = np.array(np.abs(A) > 0, dtype = int)
  return A.T

def deg_matrix(A):
  D = np.diag(np.sum(A, axis = 1))
  return D

def lap_matrix(A):
  D = deg_matrix(A)
  L = D - A
#  L = np.dot(np.sqrt(1./D), L)
#  L = np.dot(L, np.sqrt(1./D))
  return L
