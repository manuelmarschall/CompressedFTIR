import numpy as np
from scipy.linalg import svd
from scipy.sparse import (csr_matrix, coo_matrix, issparse)
from scipy.sparse import kron as spkron
from scipy.sparse import eye as speye
from scipy.sparse.linalg import spsolve

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
        float -- $\sum_{(i, j)\in\Omega} A[i, j]^2$
    """

    return np.sum([arr[x, y]**2 for (x, y) in zip(*nnz)])

def neighbors(X, Y, x, y): 
    return [(x2, y2) for x2 in range(x-1, x+2)
                            for y2 in range(y-1, y+2)
                            if (-1 < x < X and
                                -1 < y < Y and
                                (x != x2 or y != y2) and
                                (0 <= x2 < X) and
                                (0 <= y2 < Y))]


def neig_metric(u, v, X, Y):
    '''
    returns 2 if u = v, -1 if (u, v) are neighbors else 0
    '''
    
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
    # if False:
    #     # this is faster but does only work for rectangular matrices
    #     import numpy as np
    #     from scipy.sparse import lil_matrix, coo_matrix
    #     X = np.arange(0, int(n**2)).reshape([int(n), int(n)])    
    #     nrc = n**2
    #     ic = np.zeros([n+2, n+2])
    #     ic[1:-1, 1:-1] = X
        
    #     I = np.ones([n+2, n+2], dtype=np.bool)
    #     I[0, :] = 0
    #     I[-1, :] = 0
    #     I[:, 0] = 0
    #     I[:, -1] = 0
        
    #     icd = np.zeros([np.prod(n**2), 8])
    #     icd[:, 0] = ic[np.roll(I, 1, axis=1)]    # shift right
    #     icd[:, 1] = ic[np.roll(I, 1, axis=0)]    # shift down
    #     icd[:, 2] = ic[np.roll(I, -1, axis=1)]   # shift left
    #     icd[:, 3] = ic[np.roll(I, -1, axis=0)]   # shift up
        
    #     # shift up and right
    #     icd[:, 4] = ic[np.roll(np.roll(I, 1, axis=1), -1, axis=0)]
    #     # shift up and left
    #     icd[:, 5] = ic[np.roll(np.roll(I, -1, axis=1), -1, axis=0)]
    #     # shift down and right
    #     icd[:, 6] = ic[np.roll(np.roll(I, 1, axis=1), 1, axis=0)]
    #     # shift down and left
    #     icd[:, 7] = ic[np.roll(np.roll(I, -1, axis=1), 1, axis=0)]
        
    #     ic = np.tile(ic[I].reshape(-1, order="F"), (8, 1)).ravel()
    #     icd = icd.reshape(-1, order="F")
    #     data = np.ones([len(icd), 1]).ravel()
    #     Kcol_A_py = coo_matrix((data, (ic, icd)), shape=[int(n**2), int(n**2)])
    #     su = np.sum(Kcol_A_py, axis=0).T
    #     Kcol_A_py = lil_matrix(-Kcol_A_py)
    #     Kcol_A_py[2:,0] = 0
    #     Kcol_A_py[n,0] = -1
    #     Kcol_A_py[n+1,0] = -1
    #     Kcol_A_py[1, 0] = -1

    #     Kcol_A_py.setdiag(su, 0)
    #     Kcol_A_py[0, 0] = 3
    #     return Kcol_A_py

    dim_list = [n, m]
    meshgrid = np.array(np.meshgrid(*[np.arange(ndim) for ndim in dim_list]))
    X_grid = meshgrid.reshape(meshgrid.shape[0], -1).T
    dist = np.zeros([n*m]*2)
    for lia in range(n*m):
        for lib in range(n*m):
            dist[lia, lib] = neig_metric(X_grid[lia,:], X_grid[lib, :], *dim_list)
    return dist

def relative_residual(Xk, Xk1):
    return np.linalg.norm(Xk - Xk1)/np.linalg.norm(Xk)


def own_block_diag(mats, format='coo', dtype=None):
    dtype = np.dtype(dtype)
    row = []
    col = []
    data = []
    r_idx = 0
    c_idx = 0
    for a in mats:
        if issparse(a):
            a = a.tocsr()
        else:
            a = coo_matrix(a).tocsr()
        nrows, ncols = a.shape
        for r in range(nrows):
            for c in range(ncols):
                if a[r, c] is not None:
                    row.append(r + r_idx)
                    col.append(c + c_idx)
                    data.append(a[r, c])
        r_idx = r_idx + nrows
        c_idx = c_idx + ncols
    return coo_matrix((data, (row, col)), dtype=dtype).asformat(format)



def get_regularizer(n, m, t, r):
    print("get neighbor matrix")
    Kcol_A = csr_matrix(build_neighbour_matrix(int(n), int(m)))
    print("neighbor matrix done")
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
    raise NotImplemented

def curvature_lcurve(lx, ly):
    """
    Computes the curvature function of an l-curve
    TODO: calculate by hand and write in comment

    Arguments:
        lx {list} -- residual norms || Y - UV ||_\Omega
        ly {list} -- regularizer norms || lapU U || + || lapV V ||

    Returns:
        list -- discrete curvature function
    """
    dx = np.gradient(lx)
    dy = np.gradient(ly, dx)
    d2y = np.gradient(dy, dx)
    k = np.abs(d2y)/(1+dy**2)**(1.5)
    return k
    
def get_corner_node(lcurve):
    """
    Computes the optimal argument of a given l-curve.
    Note: Only approximately and discrete. 
    TODO: More elaborate using interpolation/extrapolation and 
          analytical differentiation.
          Or implement https://arxiv.org/abs/1608.04571

    Arguments:      
        lcurve {list} -- l-curve [[x1, y1], [x2, y2], ...]

    Returns:
        int -- Optimal argument that maximizes curvature
    """
    lx, ly = [], []
    for lia in range(len(lcurve)):
        lx.append(lcurve[lia][0])
        ly.append(lcurve[lia][1])
    k = curvature_lcurve(lx, ly)
    l_opt = np.argmax(k)
    return l_opt

def under_sampling(Nx, Ny, Nt, p, retries=10):
    retval = np.zeros([Nx,Ny,Nt])
    _num = int(np.ceil(np.abs(p)*Nx*Ny*Nt))
    
    curr_try = 0
    while np.any(np.sum(retval, axis=(0,1)) == 0) or np.any(np.sum(retval, axis=(0, 2)) == 0) or np.any(np.sum(retval, axis=(1, 2)) == 0):
        if curr_try > retries:
            print("unable to get enough sample points, such that every slice has data")
            break
        retval = np.zeros(Nx*Ny*Nt)
        retval[np.random.permutation(int(Nx*Ny*Nt))[:_num]] = 1
        retval = retval.reshape(Nx, Ny, Nt)
        curr_try += 1
    return retval

def subsample_3d_data(Xtrue, p):
    assert len(Xtrue.shape) == 3
    P = under_sampling(*Xtrue.shape, p)
    Xomega = np.where(P==1, Xtrue, 0)
    return Xtrue, Xomega