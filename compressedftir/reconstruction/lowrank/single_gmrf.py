'''
License

 copyright Manuel Marschall (PTB) 2020

 This software is licensed under the BSD-like license:

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in
    the documentation and/or other materials provided with the distribution.

 DISCLAIMER
 ==========
 This software was developed at Physikalisch-Technische Bundesanstalt
 (PTB). The software is made available "as is" free of cost. PTB assumes
 no responsibility whatsoever for its use by other parties, and makes no
 guarantees, expressed or implied, about its quality, reliability, safety,
 suitability or any other characteristic. In no event will PTB be liable
 for any direct, indirect or consequential damage arising in connection

Using this software in publications requires citing the following paper

Compressed FTIR spectroscopy using low-rank matrix reconstruction (to appear in Optics Express)
DOI: https://doi.org/10.1364/OE.404959
'''

import numpy as np
from scipy.linalg import svd
from scipy.sparse import (csr_matrix)
from scipy.sparse.linalg import spsolve
from compressedftir.utils import (scipy_block_diag, relative_residual, ht)
# import time


def updateU(_V, Xomega, curr_r, nnz, lmb=0.0, lap=None):
    """
    Update the spatial low-rank component.
    Alternating optimization fixes the V component of the model
    $$
      M = UV
    $$
    and aims for an optimal U such that
    $$
        U = argmin_A || Y - AV || + lmbA ||lap U||.
    $$
    This is a linear problem and can be solved directly.

    Arguments:
        _V {np.array} -- second model component
        Xomega {np.array} -- sub-sampled data
        curr_r {int} -- rank
        nnz {list} -- non zeros entries

    Keyword Arguments:
        lmb {float} -- regularization parameter (default: {0.0})
        lap {array like} -- GMRF/Laplacian regularization matrix (default: {None})

    Returns:
        array like -- new component U
    """
    hlp = []
    _n, _m = Xomega.shape
    VY = np.zeros((curr_r, _n), dtype=Xomega.dtype)
    zm = csr_matrix((curr_r, curr_r))
    # start = time.time()
    for k in range(_n):
        ind = nnz[k]
        if sum(ind) > 0:
            VO = _V[:, ind]
            hlp.append(csr_matrix(np.dot(VO, VO.T)))
            VY[:, k] = np.dot(Xomega[k, ind], VO.T)
        else:
            hlp.append(zm)
    # print("U loop: {}".format(time.time()-start))
    # start = time.time()
    H = scipy_block_diag(hlp, format="csc")
    # print("U build H: {}".format(time.time()-start))
    # print("Sum H + lap")
    if lap is not None:
        H = H + lmb*lap
    # start = time.time()
    # print("Start spsolve")
    rhs = VY.ravel(order="F")
    # print("rhs raveled")
    # TODO: Here ht?
    retval = spsolve(H, rhs).reshape((curr_r, _n), order="F").T
    # print("U solve: {}".format(time.time()-start))
    return retval


def updateV(_U, Xomega, curr_r, nnz, lmb=0.0, lap=None):
    """
    Update the second low-rank component.
    Alternating optimization fixes the U component of the model
    $$
      M = UV
    $$
    and aims for an optimal V such that
    $$
        V = argmin_A || Y - UA || + lmb ||lap V||.
    $$
    This is a linear problem and can be solved directly.

    Arguments:
        _U {np.array} -- second model component
        Xomega {np.array} -- sub-sampled data
        curr_r {int} -- rank
        nnz {list} -- non zeros entries

    Keyword Arguments:
        lmb {float} -- regularization parameter (default: {0.0})
        lap {array like} -- regularization matrix, here identity (default: {None})

    Returns:
        array like -- new component U
    """
    hlp = []
    _n, _m = Xomega.shape
    UY = np.zeros((curr_r, _m), dtype=Xomega.dtype)
    # start = time.time()
    for k in range(_m):
        ind = nnz[k]
        UO = _U[ind, :]
        hlp.append(csr_matrix(np.dot(UO.T, UO)))
        UY[:, k] = np.dot(np.transpose(Xomega[ind, k]), UO)
    # print("V loop: {}".format(time.time()-start))
    # start = time.time()
    if lap is None:
        H = scipy_block_diag(hlp, format="csr")
    else:
        H = scipy_block_diag(hlp, format="csr") + lmb*lap
    # print("V build H: {}".format(time.time()-start))
    # start = time.time()
    retval = spsolve(H, UY.ravel(order="F")).reshape((curr_r, _m), order="F")
    # print("V solve: {}".format(time.time()-start))
    return retval


def lr_recon_single(Xomega, l_regu, r, T, tau, lapU, lapV, nnz_Z0_U, nnz_Z0_V, Xtrue=None, iv_U=None, iv_V=None, bg=None):
    """
    Reconstructs a low-rank model UV to fit the data Xomega.
    Assumes a fixed rank r

    Arguments:
        Xomega {array like} -- sub-sampled data
        l_regu {float} -- regularization parameter
        r {int} -- fixed rank
        T {int} -- maximal number of iteration
        tau {float} -- tolerance
        lapU {array like} -- regularizer for U
        lapV {array like} -- regularizer for V
        nnz_Z0_U {list} -- non zero entries for U
        nnz_Z0_V {list} -- non zero entries for V

    Keyword Arguments:
        Xtrue {array like} -- Full dataset for comparison (default: {None})
        iv_U {array like} -- proposed initial value of U (default: {None})
        iv_V {array like} -- proposed initial value of V (default: {None})

    Returns:
        dict -- result dictionary containing
                U   {array like}
                V   {array like}
                resL    {list}
                resG    {list}
                if Xtrue is given: resT {list}
    """
    # normalize the data
    scl = np.std(Xomega, ddof=1)
    Xomega_scl = Xomega/scl
    # print("Do init")
    if iv_U is None or iv_V is None:
        # initialize using svd. returns (n, r)x(r,r)x(r,m)
        W, Lambda, Z = svd(Xomega_scl, full_matrices=False)
        # usually, the svd rank is larger than desired -> crop
        W = W[:, :r]
        Lambda = Lambda[:r]
        Z = Z[:r, :]
        # distribute the singular values to U and V
        U = W*Lambda**(0.5)          # shape (n, r)
        V = ht(ht(Z)*Lambda**(0.5))  # shape (r, m)
    else:
        print("initial value given")
        U = iv_U
        V = iv_V
    # initialize the residual lists
    resL = [np.infty]
    resG = [0]
    if Xtrue is not None:
        resT = []
    t = 0
    # print("Get Xt")
    Xt = np.dot(U, V) * scl
    while t < T and resL[-1] > resG[-1]*tau:
        # globstart = time.time()
        Xt_old = Xt
        # print("Update U")
        U = updateU(V, Xomega_scl, r, nnz_Z0_V, l_regu, lapU)
        # print("Update V")
        V = updateV(U, Xomega_scl, r, nnz_Z0_U, l_regu, lapV)
        # print("Compute residual")
        Xt = np.dot(U, V) * scl
        resL.append(relative_residual(Xt_old, Xt))
        resG.append(relative_residual(Xomega, Xt, check_nnz=True))
        if Xtrue is not None:
            # print("Compute residual to truth")
            resT.append(relative_residual(Xtrue, Xt + bg if bg is not None else Xt))
            print("it: {:2d}/{}, local res: {:.2e}, global res: {:.2e}, res2Full: {:.2e}".format(t+1, T, resL[-1],
                                                                                                 resG[-1], resT[-1]))
        else:
            print("it: {:2d}/{}, local res: {:.2e}, global res: {:.2e}".format(t+1, T, resL[-1], resG[-1]))
        # print("step duration: {}".format(time.time()-globstart))
        t = t+1
    retval = {
        "U": U*np.sqrt(scl),
        "V": V*np.sqrt(scl),
        "resL": resL[1:],
        "resG": resG[1:],
        "iv_U": U,
        "iv_V": V
        }
    if Xtrue is not None:
        retval["resT"] = resT
    return retval
