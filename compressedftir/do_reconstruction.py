import numpy as np
from compressedftir.utils import subsample_3d_data
from compressedftir.reconstruction.lowrank.lcurve_gmrf import do_reconstruction
from compressedftir.datareader import load_data_file

if __name__ == "__main__":
    # Xomega_test.npz will return a scipy sparse matrix of subsampled data
    # filepath = "../testdata/2d_data/Xomega_test.npz" 
    # Xomega = load_data_file(filepath)       # Get data from file. 
    filepath = "../testdata/Leishmania_Brasiliensis_High_Laser_Postion_1/original/Leishmania_Brasiliensis_Postion_1_High.csv" 
    X = load_data_file(filepath, format_hint="afm-ir")       # Get data from file. 
    # #### Data treatment
    # ###### Sparse data
    # Sometimes the data is stored as a sparse matrix, eg. as with Xomega_test.npz
    # As we do not want to deal with sparse matrices at the moment, 
    # get the full matrix. This is inefficient but for "small" problems ok.
    # Xomega = X.toarray()

    # #### 3D datacube
    # ###### Full data
    # In case the data should be subsampled to some percentage of its original size, 
    # use the following :
    p = 0.05            # reduction to 5% of data
    Xtrue, Xomega = subsample_3d_data(X, p)
    # If the data is given as a 3D datacube having (x, y, t) two spatial dimensions in front,

    # A full dataset is usually not available. 
    # If you used the subsampling, you do not need this. Otherwise, load the full data here.
    # Xtrue = None
    # Xtrue = load_data_file("../testdata/2d_data/Xtrue.npy")

    # Where the results should be exported
    exportpath = "../testdata/Leishmania_Brasiliensis_High_Laser_Postion_1/samplerun/"            
    export_every_lambda_result = True       # Flag to export every l-curve value result

    r = 5                                   # desired rank parameter
    tau = 1e-5                              # convergence tolerance
    lam = np.logspace(-8, 1, num=100)       # L-curve points
    max_iter = 200                         # maximal number of local interation


    do_reconstruction(Xomega, r, lam, tau, max_iter, exportpath, export_every_lambda_result, Xtrue)