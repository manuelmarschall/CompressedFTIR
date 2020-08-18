import numpy as np
from scipy.linalg import svd
from scipy.sparse import (csr_matrix, coo_matrix, issparse)
from scipy.sparse import kron as spkron
from scipy.sparse import eye as speye
from scipy.sparse.linalg import spsolve
from compressedftir.utils import (own_block_diag, relative_residual)

def updateU(_V, Xomega, curr_r, nnz, lmbA=0.0, lap=None):
    hlp = []
    _n, _m = Xomega.shape
    VY = np.zeros((curr_r, _n))
    zm = csr_matrix((curr_r, curr_r))
    for k in range(_n):
        ind = nnz[k]
        if sum(ind)>0:
            VO = _V[:, ind]
            hlp.append(csr_matrix(np.dot(VO, VO.T)))
            VY[:, k] = np.dot(Xomega[k, ind], VO.T)
        else:
            hlp.append(zm)
    H = own_block_diag(hlp, format="csr")
    if lap is not None:
        H = H + lmbA*lap
    return spsolve(H, VY.ravel(order="F")).reshape((curr_r, _n), order="F").T

def updateV(_U, Xomega, curr_r, nnz, lmb=0.0, lap=None):
    hlp = []
    _n, _m = Xomega.shape
    UY = np.zeros((curr_r, _m))
    zm = csr_matrix((curr_r, curr_r))
    for k in range(_m):
        ind = nnz[k]
        UO = _U[ind, :]
        hlp.append(csr_matrix(np.dot(UO.T, UO)))
        UY[:, k] = np.dot(Xomega[ind, k].T, UO)
        
    if lap is None:
        H = own_block_diag(hlp, format="csr")
    else:
        H = own_block_diag(hlp, format="csr") + lmb*lap

    return spsolve(H, UY.ravel(order="F")).reshape((curr_r, _m), order="F")

def lr_recon_single(Xomega, l_regu, r, T, tau, lapU, lapV, nnz_Z0_U, nnz_Z0_V, Xtrue=None):
    
    scl = np.std(Xomega, ddof=1)
    Xomega_scl = Xomega/scl
    # initialize using svd. returns (n, r)x(r,r)x(r,m)
    W, Lambda, Z = svd(Xomega_scl, full_matrices=False)
    # usually, the svd rank is larger than desired -> crop
    W = W[:, :r]; Lambda = Lambda[:r]; Z = Z[:r, :]

    U = W*Lambda**(0.5)          # shape (n, r)
    V = (Z.T*Lambda**(0.5)).T    # shape (r, m)
    resL = [np.infty]
    resG = [0]
    if Xtrue is not None:
        resT = []
    t = 0
    Xt = np.dot(U, V) * scl
    while t < T and resL[-1] > resG[-1]*tau:
        Xt_old = Xt
        U = updateU(V, Xomega_scl, r, nnz_Z0_V, l_regu, lapU)
        V = updateV(U, Xomega_scl, r, nnz_Z0_U, l_regu, lapV)
        Xt = np.dot(U, V) * scl
        resL.append(relative_residual(Xt_old, Xt))
        resG.append(relative_residual(Xomega, Xt))
        if Xtrue is not None:
            resT.append(relative_residual(Xtrue, Xt))
            print("it: {:2d}/{}, local res: {:.2e}, global res: {:.2e}, res2Full: {:.2e}".format(t+1, T, resL[-1], resG[-1], resT[-1]))
        else:    
            print("it: {:2d}/{}, local res: {:.2e}, global res: {:.2e}".format(t+1, T, resL[-1], resG[-1]))
        t = t+1
    retval = {
        "U": U*np.sqrt(scl),
        "V": V*np.sqrt(scl),
        "resL": resL[1:],
        "resG": resG[1:]
        }
    if Xtrue is not None:
        retval["resT"] = resT
    return retval