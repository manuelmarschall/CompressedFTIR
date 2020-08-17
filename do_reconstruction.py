import numpy as np
from scipy.sparse import load_npz
from utils import (lr_recon_single, get_regularizer, get_nnz_indices)
import matplotlib.pyplot as plt

filepath = "Xomega_test.npz"            # sparse samples dataset

Xomega = load_npz(filepath)             # assumes scipy sparse matrix

r = 5                                   # desired rank parameter
tau = 1e-5                              # convergence tolerance
lam = np.logspace(-4, -1, num=10)       # L-curve points
max_iter = 50                          # maximal number of local interation

Xh = dict()
res_l = np.zeros(len(lam))
res_g = np.zeros(len(lam))
lcurve = dict()

# data-preprocessing
Z0 = Xomega.toarray()
scl = np.std(Z0, ddof=1)
Xomega = Xomega/scl
nnz = np.nonzero(Z0)

def sum_sq_nnz(nnz, arr):
    return np.sum([arr[x, y]**2 for (x, y) in zip(*nnz)])

y2nrm = sum_sq_nnz(nnz, Z0)

# solver preparation
lapU, lapV = get_regularizer(np.sqrt(Xomega.shape[0]), Xomega.shape[1], r)
nnzU, nnzV = get_nnz_indices(Xomega)
solver_info = {
    "Xomega": Xomega,
    "r": r,
    "T": max_iter,
    "tau": tau,
    "lapU": lapU,
    "lapV": lapV,
    "nnz_Z0_U": nnzU,
    "nnz_Z0_V": nnzV
}

# start iteration over L-curve items
for lia, l in enumerate(lam):
    print("L-curve step: {}/{}".format(lia+1, len(lam)))
    print("Experimental setup:\n lambda: {}\n maximal rank: {}\n convergence tol: {}\n max iteration: {}".format(l, r, tau, max_iter))
    solver_info["l_regu"] = l
    curr_result = lr_recon_single(**solver_info)
    Xh[l] = curr_result["Xh"] * scl
    res_l[lia] = curr_result["resL"]
    res_g[lia] = curr_result["resG"]
    U = curr_result["U"] * np.sqrt(scl)
    V = curr_result["V"] * np.sqrt(scl)
    dd = Xh[l] - Z0
    lcurve[lia] = [sum_sq_nnz(nnz, dd)/y2nrm, U.reshape(-1, order="F").dot(lapU.dot(U.reshape(-1, order="F"))) + V.reshape(-1, order="F").dot(lapV.dot(V.reshape(-1, order="F")))]



fig = plt.figure()
plt.plot([np.log(lcurve[lia][0]) for lia in range(len(lam))], [np.log(lcurve[lia][1]) for lia in range(len(lam))])
plt.show()