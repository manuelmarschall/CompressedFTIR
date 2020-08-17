import numpy as np
from scipy.linalg import svd
from scipy.sparse import (csr_matrix, coo_matrix, issparse, load_npz)
from scipy.sparse import kron as spkron
from scipy.sparse import eye as speye
from scipy.sparse.linalg import (spsolve)

def load_data_file(filepath):
    # TODO: add csv AFM data
    assert isinstance(filepath, str)
    # check for file suffix. Can be done more elaborate
    if np.char.endswith(filepath, ".npz"):
        # scipy sparse format or numpy compressed
        try:
            retval = load_npz(filepath)
        except IOError:
            try:
                retval = np.load(filepath)
            except IOError as exception:
                print("File does not exists, or ends with .npz and is neither a scipy sparse matrix or in compressed numpy format.")
                print(exception)
                print(" exit!")
                exit()
        
    elif np.char.endswith(filepath, ".npy"):
        # numpy matrix format
        try:
            retval = np.load(filepath)
        except IOError as exception:
            print("File does not exists or cannot be read")
            print(exception)
            print(" exit!")
            exit()
        except ValueError as exception:
            print("The file contains an object array, but allow_pickle=False given")
            print(exception)
            print(" exit!")
            exit()
    else:
        raise ValueError("Unknown suffix. exit!")
    return retval

def build_neighbour_matrix(n):
    import numpy as np
    from scipy.sparse import lil_matrix, coo_matrix
    X = np.arange(0, int(n**2)).reshape([int(n), int(n)])    
    nrc = n**2
    ic = np.zeros([n+2, n+2])
    ic[1:-1, 1:-1] = X
    
    I = np.ones([n+2, n+2], dtype=np.bool)
    I[0, :] = 0
    I[-1, :] = 0
    I[:, 0] = 0
    I[:, -1] = 0
    
    icd = np.zeros([np.prod(n**2), 8])
    icd[:, 0] = ic[np.roll(I, 1, axis=1)]    # shift right
    icd[:, 1] = ic[np.roll(I, 1, axis=0)]    # shift down
    icd[:, 2] = ic[np.roll(I, -1, axis=1)]   # shift left
    icd[:, 3] = ic[np.roll(I, -1, axis=0)]   # shift up
    
    # shift up and right
    icd[:, 4] = ic[np.roll(np.roll(I, 1, axis=1), -1, axis=0)]
    # shift up and left
    icd[:, 5] = ic[np.roll(np.roll(I, -1, axis=1), -1, axis=0)]
    # shift down and right
    icd[:, 6] = ic[np.roll(np.roll(I, 1, axis=1), 1, axis=0)]
    # shift down and left
    icd[:, 7] = ic[np.roll(np.roll(I, -1, axis=1), 1, axis=0)]
    
    ic = np.tile(ic[I].reshape(-1, order="F"), (8, 1)).ravel()
    icd = icd.reshape(-1, order="F")
    data = np.ones([len(icd), 1]).ravel()
    Kcol_A_py = coo_matrix((data, (ic, icd)), shape=[int(n**2), int(n**2)])
    su = np.sum(Kcol_A_py, axis=0).T
    Kcol_A_py = lil_matrix(-Kcol_A_py)
    Kcol_A_py[2:,0] = 0
    Kcol_A_py[n,0] = -1
    Kcol_A_py[n+1,0] = -1
    Kcol_A_py[1, 0] = -1

    Kcol_A_py.setdiag(su, 0)
    Kcol_A_py[0, 0] = 3
    return Kcol_A_py

def own_block_diag(mats, format='coo', dtype=None):
    dtype = np.dtype(dtype)
    row = []
    col = []
    data = []
    r_idx = 0
    c_idx = 0
    for ia, a in enumerate(mats):
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

def updateV(_A, Xomega, curr_r, nnz, lmb=0.0, lap=None):
    hlp = []
    _n, _m = Xomega.shape
    AY = np.zeros((curr_r, _m))
    zm = csr_matrix((curr_r, curr_r))
    for k in range(_m):
        ind = nnz[k]
        AO = _A[ind, :]
        hlp.append(csr_matrix(np.dot(AO.T, AO)))
        AY[:, k] = np.dot(Xomega[ind, k].T, AO)
        
    if lap is None:
        H = own_block_diag(hlp, format="csr")
    else:
        H = own_block_diag(hlp, format="csr") + lmb*lap

    return spsolve(H, AY.ravel(order="F")).reshape((curr_r, _m), order="F")

def reslocal(Xk, Xk1):
    return np.linalg.norm(Xk - Xk1)/np.linalg.norm(Xk1)
def resglobal(Xk, Xomega):
    return np.linalg.norm(Xomega-Xk)/np.linalg.norm(Xomega)
def lr_recon_single(Xomega, l_regu, r, T, tau, lapU, lapV, nnz_Z0_U, nnz_Z0_V):
    W, Lambda, Z = svd(Xomega, full_matrices=False)
    U = W*Lambda**(0.5)          # shape (n, r)
    V = (Z.T*Lambda**(0.5)).T    # shape (r, m)
    resL = np.infty
    resG = 0
    t = 0
    Xt = np.dot(U, V)
    while t < T and resL > resG*tau:
        Xt_old = Xt
        U = updateU(V, Xomega, r, nnz_Z0_V, l_regu, lapU)
        V = updateV(U, Xomega, r, nnz_Z0_U, l_regu, lapV)
        Xt = np.dot(U, V)
        resL = reslocal(Xt, Xt_old)
        resG = resglobal(Xt, Xomega)
        print("it: {}/{}, local res: {}, global res: {}".format(t+1, T, resL, resG))
        t = t+1
    retval = {
        "Xh": Xt,
        "U": U,
        "V": V,
        "resL": resL,
        "resG": resG
        }
    return retval

def get_regularizer(n, m, r):
    Kcol_A = csr_matrix(build_neighbour_matrix(int(n)))
    Kcol_B = speye(m)
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