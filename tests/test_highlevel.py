from compressedftir.reconstruction.lowrank.lcurve_gmrf import do_reconstruction
import numpy as np
import json


def test_recon():
    Xomega = np.random.normal(np.eye(9), 1e-6)

    Xtrue = np.eye(Xomega.shape[0])
    # Note where the results should be exported to
    exportpath = "testdata/2d_data/tests/"
    export_every_lambda_result = True       # Flag to export every l-curve value result

    r = 3                                   # desired rank parameter
    tau = 1e-2                              # convergence tolerance
    # L-curve points: WARNING: should be a numpy array atm
    lam = np.flip(np.logspace(-5, 1, num=10))
    max_iter = 20                          # maximal number of local interation

    do_reconstruction(Xomega, r, lam, tau, max_iter, exportpath, export_every_lambda_result, Xtrue, load=False)
    with open(exportpath + "result.dat") as fp:
        res = json.load(fp)
    Xh = np.dot(res["U"], res["V"])
    print(Xh)
    err = np.linalg.norm(Xh - Xtrue)
    assert err < 1e-6
