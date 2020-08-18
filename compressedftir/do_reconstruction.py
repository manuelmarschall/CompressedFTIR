import numpy as np
from compressedftir.utils import (get_regularizer, get_nnz_indices, sum_sq_nnz, get_corner_node)
from compressedftir.reconstruction.lowrank.lcurve_gmrf import do_reconstruction
from compressedftir.datareader import load_data_file

# Xomega_test.npz will return a scipy sparse matrix of subsampled data
filepath = "../testdata/2d_data/Xomega_test.npz" 
Xomega = load_data_file(filepath)       # Get data from file. 
# As we do not want to deal with sparse matrices at the moment, 
# get the full matrix. This is inefficient but for "small" problems ok.
# If load_data_file returns a full matrix, comment out this line
Xomega = Xomega.toarray()

# A full dataset is usually not available. But if, load it here.
# Xtrue = None
Xtrue = load_data_file("../testdata/2d_data/Xtrue.npy")

exportpath = "test_result2/"            # Where the results should be exported
export_every_lambda_result = True       # Flag to export every l-curve value result

r = 5                                   # desired rank parameter
tau = 1e-5                              # convergence tolerance
lam = np.logspace(-8, 1, num=100)       # L-curve points
max_iter = 50                           # maximal number of local interation


do_reconstruction(Xomega, r, lam, tau, max_iter, exportpath, export_every_lambda_result, Xtrue)