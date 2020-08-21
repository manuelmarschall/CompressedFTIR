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
    Xomega = X.toarray()
    # A full dataset is usually not available. 
    # If you used the subsampling, you do not need this. Otherwise, load the full data here.
    # Xtrue = None
    Xtrue = load_data_file("testdata/2d_data/Xtrue.npy")

    # Note where the results should be exported to
    exportpath = "testdata/2d_data/samplerun/"            
    export_every_lambda_result = True       # Flag to export every l-curve value result
    
    r = 5                                   # desired rank parameter
    tau = 1e-5                              # convergence tolerance
    lam = np.logspace(-8, 1, num=100)       # L-curve points: WARNING: should be a numpy array atm
    max_iter = 200                         # maximal number of local interation


    do_reconstruction(Xomega, r, lam, tau, max_iter, exportpath, export_every_lambda_result, Xtrue, load=False)


def runLeishmania_AFM_IR_test():
    filepath = "testdata/Leishmania_Brasiliensis_Low_Laser_Postion_1/original/Leishmania_Brasiliensis_Postion_1_Low.csv" 
    Xtrue = load_data_file(filepath, format_hint="afm-ir")       # Get data from file. 
    # #### Data treatment
    
    # #### 3D datacube
    # ###### Full data
    # In case the data should be subsampled to some percentage of its original size, 
    # use the following :
    p = 0.15            # reduction to 15% of data
    Xomega = subsample_3d_data(Xtrue, p)
    # the data is given as a 3D datacube having (x, y, t) two spatial dimensions (x,y) and
    # a measurement dependent third dimension
    
    # Note where the results should be exported to
    exportpath = "testdata/Leishmania_Brasiliensis_Low_Laser_Postion_1/samplerun_015p/"            
    export_every_lambda_result = True       # Flag to export every l-curve value result
    r = 5                                   # desired rank parameter
    tau = 1e-5                              # convergence tolerance
    lam = np.logspace(-8, 1, num=100)       # L-curve points: WARNING: should be a numpy array atm
    max_iter = 200                         # maximal number of local interation


    do_reconstruction(Xomega, r, lam, tau, max_iter, exportpath, export_every_lambda_result, Xtrue, load=False)

def runLeishmania_FPA_test():
    filepath = "testdata/Leishmania_Brasiliensis_FPA_4/2020-01-FPALeish_4.mat" 
    print("Load dataset")
    Xtrue = load_data_file(filepath, format_hint="leishmania-fpa")       # Get data from file. 
    # #### Data treatment
    
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
    exportpath = "testdata/Leishmania_Brasiliensis_FPA_4/samplerun_015p/"            
    export_every_lambda_result = True       # Flag to export every l-curve value result
    r = 5                                   # desired rank parameter
    tau = 1e-5                              # convergence tolerance
    lam = np.logspace(-3, 0, num=30)       # L-curve points: WARNING: should be a numpy array atm
    max_iter = 500                         # maximal number of local interation


    do_reconstruction(Xomega, r, lam, tau, max_iter, exportpath, export_every_lambda_result, Xtrue, load=False)

if __name__ == "__main__":
    # run2Dtest()
    # runLeishmania_AFM_IR_test()


    runLeishmania_FPA_test()

    # The results are now stored in the defined 'exportpath' and can be read from disc
    # using json. 
    # The low-rank reconstruction at the optimal value is stored in the keys "U" and "V".
    # Use matrix multiplication UV to obtain the low-rank approximation to your data.
    
    