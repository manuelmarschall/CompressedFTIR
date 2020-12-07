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
from compressedftir.utils import subsample_3d_data
from compressedftir.reconstruction.lowrank.lcurve_gmrf import do_reconstruction
from compressedftir.datareader import load_data_file


def run2Dtest():
    # Xomega_test.npz will return a scipy sparse matrix of subsampled data
    filepath = "testdata/2d_data/Xomega_test.npz"
    X = load_data_file(filepath)       # Get data from file.

    # ###### Sparse data
    # Sometimes the data is stored as a sparse matrix, eg. as with Xomega_test.npz
    # As we do not want to deal with sparse matrices at the moment,
    # get the full matrix. This is inefficient but for "small" problems ok.
    X.data = X.data + np.random.randn(len(X.data))*0.1
    Xomega = X.toarray()

    # A full dataset is usually not available.
    # If you used the subsampling, you do not need this. Otherwise, load the full data here.
    # Xtrue = None
    Xtrue = load_data_file("testdata/2d_data/Xtrue.npy")
    # Note where the results should be exported to
    exportpath = "testdata/2d_data/samplerun/"
    export_every_lambda_result = True       # Flag to export every l-curve value result

    r = 5                                   # desired rank parameter
    tau = 1e-2                              # convergence tolerance
    # L-curve points: WARNING: should be a numpy array atm
    lam = np.flip(np.logspace(-5, 1, num=10))
    max_iter = 20                          # maximal number of local interation

    do_reconstruction(Xomega, r, lam, tau, max_iter, exportpath, export_every_lambda_result, Xtrue, load=False)


def runLeishmania_AFM_IR_test():
    filepath = "testdata/Leishmania_Brasiliensis_Low_Laser_Postion_1/original/Leishmania_Brasiliensis_Postion_1_Low.csv"
    Xtrue = load_data_file(filepath, format_hint="afm-ir")       # Get data from file.
    # #### Data treatment

    # #### 3D datacube
    # ###### Full data
    # In case the data should be subsampled to some percentage of its original size,
    # use the following :
    p = 0.10            # reduction to 15% of data
    Xomega = subsample_3d_data(Xtrue, p)
    # the data is given as a 3D datacube having (x, y, t) two spatial dimensions (x,y) and
    # a measurement dependent third dimension

    # Note where the results should be exported to
    exportpath = "testdata/Leishmania_Brasiliensis_Low_Laser_Postion_1/samplerun_010p/"
    export_every_lambda_result = True       # Flag to export every l-curve value result
    r = 5                                   # desired rank parameter
    tau = 1e-2                              # convergence tolerance
    # L-curve points: WARNING: should be a numpy array atm
    lam = np.flip(np.logspace(-5, 1, num=10))
    max_iter = 20                         # maximal number of local interation

    do_reconstruction(Xomega, r, lam, tau, max_iter, exportpath, export_every_lambda_result, Xtrue, load=False)


def runLeishmania_FPA_test():
    # filepath = "testdata/Leishmania_Brasiliensis_FPA_4/2020-01-FPALeish_4.mat"
    filepath = "testdata/Leishmania_Ltar/2020-01-FPALeish_1.mat"
    print("Load dataset")
    Xtrue = load_data_file(filepath, format_hint="leishmania-fpa")       # Get data from file.
    # #### Data treatment

    # we substract an offset that interrupts the reconstruction.
    # TODO: save the offset and add it to the reconstruction afterwards
    bg = np.repeat(np.mean(Xtrue, axis=2), Xtrue.shape[2]).reshape(*Xtrue.shape)
    Xtrue = Xtrue - bg

    # #### 3D datacube
    # ###### Full data
    # In case the data should be subsampled to some percentage of its original size,
    # use the following :
    print("subsample Full dataset")
    p = 0.15            # reduction to 15% of data
    Xomega = subsample_3d_data(Xtrue, p)
    # the data is given as a 3D datacube having (x, y, t) two spatial dimensions (x,y) and
    # a measurement dependent third dimension
    
    # Note where the results should be exported to
    exportpath = "testdata/Leishmania_Ltar/samplerun_015p/"
    export_every_lambda_result = True       # Flag to export every l-curve value result
    r = 20                                   # desired rank parameter
    tau = 1e-2                              # convergence tolerance
    # L-curve points: WARNING: should be a numpy array atm
    # lam = np.flip(np.logspace(-4, -1, num=20))[8]
    lam = np.flip(np.logspace(-4, -1, num=20))
    # lam = np.array([lam])
    max_iter = 30                          # maximal number of local interation

    do_reconstruction(Xomega, r, lam, tau, max_iter, exportpath, export_every_lambda_result, Xtrue, load=False)


if __name__ == "__main__":
    # run2Dtest()
    runLeishmania_AFM_IR_test()

    # The results are now stored in the defined 'exportpath' and can be read from disc
    # using json.
    # The low-rank reconstruction at the optimal value is stored in the keys "U" and "V".
    # Use matrix multiplication UV to obtain the low-rank approximation to your data.
