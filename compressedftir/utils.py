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
DOI: ???
'''
import numpy as np
# from scipy.linalg import svd
from scipy.sparse import (csr_matrix, coo_matrix, issparse, lil_matrix)
from scipy.sparse import kron as spkron
from scipy.sparse import eye as speye
# from scipy.sparse.linalg import spsolve


def stop_at_exception(ex, err_str=None):
    """
    Small wrapper for exceptions and termination

    Arguments:
        ex {Exception} -- raised by code

    Keyword Arguments:
        err_str {str} -- String to write before expection and exit (default: {None})
    """
    if err_str is not None:
        print(err_str)
    print(ex)
    print(" exit!")
    exit()


def sum_sq_nnz(nnz, arr):
    """
    Returns the sum of squares using non zero mapping

    Arguments:
        nnz {list} -- list of non-zero indices
        arr {np.array} -- 2D array

    Returns:
        float -- $sum{(i, j) in Omega} A[i, j]^2$
    """

    return np.sum([arr[x, y]**2 for (x, y) in zip(*nnz)])


def neighbors(X, Y, x, y):
    """
    returns a list of tuples containing all neighbor
    of the vertex (x, y)

    Arguments:
        X {int} -- number of vertices in X direction
        Y {int} -- number of vertices in Y direction
        x {int} -- x position of vertex 0 <= x <= X
        y {int} -- y position of vertex 0 <= y <= Y

    Returns:
        list -- neighbor of (x, y)
    """
    return [(x2, y2) for x2 in range(x-1, x+2)
            for y2 in range(y-1, y+2)
            if (-1 < x < X and
                -1 < y < Y and
                (x != x2 or y != y2) and
                (0 <= x2 < X) and
                (0 <= y2 < Y))]


def neig_metric(u, v, X, Y):
    """
    Metric for neighbor vertices in a regular matrix graph

    Arguments:
        u {list} -- vertex 1
        v {list} -- vertex 2
        X {int} -- number of vertices in X direction
        Y {int} -- number of vertices in Y direction

    Returns:
        int -- 2 if u = v, -1 if (u, v) are neighbors else 0
    """

    neigu = neighbors(X, Y, u[0], u[1])
    if np.abs(u - v).sum() == 1:
        return -1
    elif np.abs(u-v).sum() == 0:
        return len(neigu)
    elif np.abs(u - v).sum() == 2:
        for nu in neigu:
            if nu == tuple(v):
                return -1
    return 0


def build_neighbour_matrix(n, m):
    """
    Generates a matrix of size (n*m x n*m) indicating the number of neighbor
    of a pixel by calling neig_metric for every pixel/ vertex combination

    Arguments:
        n {int} -- number of vertices in x direction
        m {int} -- number of vertices in y direction

    Returns:
        sparse lil-matrix -- neighbor matrix
    """

    X = np.arange(0, int(n*m)).reshape([int(n), int(m)], order="F")
    # nrc = n*m
    ic = np.zeros([n+2, m+2])
    ic[1:-1, 1:-1] = X

    Ind = np.ones([n+2, m+2], dtype=np.bool)
    Ind[0, :] = 0
    Ind[-1, :] = 0
    Ind[:, 0] = 0
    Ind[:, -1] = 0

    icd = np.zeros([np.prod(n*m), 8])
    icd[:, 0] = ic[np.roll(Ind, 1, axis=1)]    # shift right
    icd[:, 1] = ic[np.roll(Ind, 1, axis=0)]    # shift down
    icd[:, 2] = ic[np.roll(Ind, -1, axis=1)]   # shift left
    icd[:, 3] = ic[np.roll(Ind, -1, axis=0)]   # shift up

    # shift up and right
    icd[:, 4] = ic[np.roll(np.roll(Ind, 1, axis=1), -1, axis=0)]
    # shift up and left
    icd[:, 5] = ic[np.roll(np.roll(Ind, -1, axis=1), -1, axis=0)]
    # shift down and right
    icd[:, 6] = ic[np.roll(np.roll(Ind, 1, axis=1), 1, axis=0)]
    # shift down and left
    icd[:, 7] = ic[np.roll(np.roll(Ind, -1, axis=1), 1, axis=0)]

    ic = np.tile(ic[Ind].reshape(-1, order="F"), (8, 1)).ravel(order="C")
    icd = icd.reshape(-1, order="F")
    data = np.ones([len(icd), 1]).ravel(order="F")
    Kcol_A_py = coo_matrix((data, (ic, icd)), shape=[int(n*m), int(n*m)])
    su = np.sum(Kcol_A_py, axis=0).T
    Kcol_A_py = lil_matrix(-Kcol_A_py)
    Kcol_A_py[2:, 0] = 0
    Kcol_A_py[n, 0] = -1
    Kcol_A_py[n+1, 0] = -1
    Kcol_A_py[1, 0] = -1

    Kcol_A_py.setdiag(su, 0)
    Kcol_A_py[0, 0] = 3
    dist = Kcol_A_py

    # This is the old, slow but safe version. Really iterating over everything
    # if cache:
    #     import os
    #     if not os.path.exists("compFTIRtmp/neighborMat/"):
    #         os.makedirs("compFTIRtmp/neighborMat/")
    #     curr_mat = "compFTIRtmp/neighborMat/{}{}".format(n,m)
    #     if os.path.exists(curr_mat + ".npy"):
    #         return np.load(curr_mat + ".npy")
    # dim_list = [n, m]
    # meshgrid = np.array(np.meshgrid(*[np.arange(ndim) for ndim in dim_list]))
    # X_grid = meshgrid.reshape(meshgrid.shape[0], -1).T
    # dist = np.zeros([n*m]*2)
    # for lia in range(n*m):
    #     for lib in range(n*m):
    #         dist[lia, lib] = neig_metric(X_grid[lia,:], X_grid[lib, :], *dim_list)
    # if cache:
    #     np.save(curr_mat, dist)

    # assert np.linalg.norm(Kcol_A_py - dist) <= 1e-10

    return dist


def relative_residual(Xk, Xk1, check_nnz=False):
    """
    computes a relative residual of two matrices in the default norm
    $$
        || Xk - Xk1 || / || Xk ||
    $$

    Arguments:
        Xk {array like} -- matrix 1
        Xk1 {array like} -- matrix 2

    Returns:
        float -- relative residual
    """
    if check_nnz:
        nnz = np.nonzero(Xk)
        retval = sum_sq_nnz(nnz, Xk-Xk1)
        retval /= sum_sq_nnz(nnz, Xk)
        return retval
    return np.linalg.norm(Xk - Xk1)/np.linalg.norm(Xk)


def scipy_block_diag(mats, format=None, dtype=None):
    """
    This is taken from a new version of scipy related to the issue
    https://github.com/scipy/scipy/pull/12356

    Build a block diagonal sparse matrix from provided matrices.
    Parameters
    ----------
    mats : sequence of matrices
        Input matrices.
    format : str, optional
        The sparse format of the result (e.g., "csr"). If not given, the matrix
        is returned in "coo" format.
    dtype : dtype specifier, optional
        The data-type of the output matrix. If not given, the dtype is
        determined from that of `blocks`.
    Returns
    -------
    res : sparse matrix
    Notes
    -----
    .. versionadded:: 0.11.0
    See Also
    --------
    bmat, diags
    Examples
    --------
    >>> from scipy.sparse import coo_matrix, block_diag
    >>> A = coo_matrix([[1, 2], [3, 4]])
    >>> B = coo_matrix([[5], [6]])
    >>> C = coo_matrix([[7]])
    >>> block_diag((A, B, C)).toarray()
    array([[1, 2, 0, 0],
           [3, 4, 0, 0],
           [0, 0, 5, 0],
           [0, 0, 6, 0],
           [0, 0, 0, 7]])
    """
    row = []
    col = []
    data = []
    r_idx = 0
    c_idx = 0
    for a in mats:
        nrows, ncols = a.shape
        if issparse(a):
            a = a.tocoo()
            row.append(a.row + r_idx)
            col.append(a.col + c_idx)
            data.append(a.data)
        else:
            a_row, a_col = np.divmod(np.arange(nrows*ncols), ncols)
            row.append(a_row + r_idx)
            col.append(a_col + c_idx)
            data.append(a.ravel())
        r_idx += nrows
        c_idx += ncols
    row = np.concatenate(row)
    col = np.concatenate(col)
    data = np.concatenate(data)
    return coo_matrix((data, (row, col)),
                      shape=(r_idx, c_idx),
                      dtype=dtype).asformat(format)


def get_regularizer(n, m, t, r):
    Kcol_A = csr_matrix(build_neighbour_matrix(int(n), int(m)))
    Kcol_B = speye(t)
    # build regularization matrices
    lapU = spkron(Kcol_A, speye(r))
    lapV = spkron(Kcol_B, speye(r))
    return lapU, lapV


def get_nnz_indices(Xomega):
    """
    Get two lists of non zero indices; row-wise and column-wise.
    This is highly inefficient and probably can be done faster using
    csr/csc sparse formats.
    """
    nnz_Z0_U = []                           # for faster indexing later
    nnz_Z0_V = []                           # get row and column indices
    for lia in range(Xomega.shape[1]):
        nnz_Z0_U.append(np.nonzero(Xomega[:, lia])[0])

    for lia in range(Xomega.shape[0]):
        nnz_Z0_V.append(np.nonzero(Xomega[lia, :])[0])

    return nnz_Z0_U, nnz_Z0_V


def serialize_coo_matrix(spmat):
    """
    Generate a json serializeable object from a sparse (coo) matrix

    Arguments:
        spmat {scipy.sparse.coo_matrix} -- sparse matrix in coordinate format
    """
    # TODO: get i, j, v and return list[i, j, v]
    raise NotImplementedError


def curvature_lcurve(lx, ly):
    """
    Computes the curvature function of an l-curve
    TODO: calculate by hand and write in comment

    Arguments:
        lx {list} -- residual norms || Y - UV ||_Omega
        ly {list} -- regularizer norms || lapU U || + || lapV V ||

    Returns:
        list -- discrete curvature function
    """
    dx = np.gradient(lx)
    dy = np.gradient(ly, dx)
    d2y = np.gradient(dy, dx)
    k = np.abs(d2y)/((1+dy**2)**(1.5))
    return k


def under_sampling(Nx, Ny, Nt, p, retries=10):
    """
    Generates a mask having (100*p)% of nodes active.
    The nodes are chosen uniformly in the given dimensions.

    TODO: This is bad style. Use a generic dimension list instead

    Arguments:
        Nx {int} -- first dimension
        Ny {int} -- second dimension
        Nt {int} -- third dimension
        p {float} -- 1/100 percentage of data

    Keyword Arguments:
        retries {int} -- repetitions in case the sampling creates something odd (default: {10})

    Returns:
        array like -- mask - 3D tensor containing 1 and 0
    """
    retval = np.zeros([Nx, Ny, Nt])
    _num = int(np.ceil(np.abs(p)*Nx*Ny*Nt))

    curr_try = 0
    while np.any(np.sum(retval, axis=(0, 1)) == 0) or np.any(np.sum(retval, axis=(0, 2)) == 0) \
            or np.any(np.sum(retval, axis=(1, 2)) == 0):
        if curr_try > retries:
            print("unable to get enough sample points, such that every slice has data")
            break
        retval = np.zeros(Nx*Ny*Nt)
        retval[np.random.permutation(int(Nx*Ny*Nt))[:_num]] = 1
        retval = retval.reshape(Nx, Ny, Nt)
        curr_try += 1
    return retval


def subsample_3d_data(Xtrue, p):
    """
    Generates a sub-sampled data cube having uniformly chosen 100p % of data

    Arguments:
        Xtrue {array like} -- full dataset
        p {float} -- 1/100 percentage of data

    Returns:
        array like -- Full dataset and sub-samples dataset
    """
    assert len(Xtrue.shape) == 3
    P = under_sampling(*Xtrue.shape, p)
    Xomega = np.where(P == 1, Xtrue, 0)
    return Xomega
