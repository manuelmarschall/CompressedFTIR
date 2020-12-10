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
from alea.utils.progress.percentage import PercentageBar
from compressedftir.datareader import load_data_file
import json
from math import log10, sqrt

from skimage.metrics import structural_similarity as compare_ssim

def ssi(original, compressed):

    # 5. Compute the Structural Similarity Index (SSIM) between the two
    #    images, ensuring that the difference image is returned
    score, _ = compare_ssim(original, compressed, full=True)
    # diff = (diff * 255).astype("uint8")
    return score

def SSITensor(X, Xh):
    ssi_list = []
    for lia in range(X.shape[2]):
        ssi_list.append(ssi(X[:, :, lia], Xh[:, :, lia]))
    return np.mean(np.array(ssi_list))

def PSNRTensor(X, Xh):
    psnr_list = []
    for lia in range(X.shape[2]):
        psnr_list.append(PSNR(X[:, :, lia], Xh[:, :, lia]))
    return np.mean(np.array(psnr_list))

def PSNR(original, compressed): 
    mse = MSE(original, compressed)
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
                  # Therefore PSNR have no importance. 
        return 100
    max_pixel = np.max(original)  #255.0
    psnr = 20 * log10(max_pixel / sqrt(mse)) 
    return psnr 

def MSE(original, compressed):
    return np.mean((original - compressed) ** 2) 

dataset = "Leishmania 1"
filepath = "testdata/Leishmania_Ltar/2020-01-FPALeish_1.mat" 
print("Load dataset")
Xtrue = load_data_file(filepath, format_hint="leishmania-fpa")       # Get data from file. 
n, m = Xtrue.shape[0], Xtrue.shape[1]

bg = np.repeat(np.mean(Xtrue, axis=2), Xtrue.shape[2]).reshape(*Xtrue.shape)

print("read approximation result")
paths = [
    "testdata/Leishmania_Ltar/samplerun_001p/iter10/",
    "testdata/Leishmania_Ltar/samplerun_005p/iter6/",
    "testdata/Leishmania_Ltar/samplerun_010p/iter5/",
    "testdata/Leishmania_Ltar/samplerun_015p/iter5/",
    "testdata/Leishmania_Ltar/samplerun_050p/iter2/"
]

Xh_list = []
for curr_path in paths:
    with open(curr_path + "result.dat", "r") as fp:
        res = json.load(fp)
    U, V = np.array(res["U"]), np.array(res["V"])
    Xh_list.append(np.dot(U, V).reshape(*Xtrue.shape) + bg)
    print("directory: {}".format(curr_path))
    print("  Mean PSNR: {} dB".format(PSNRTensor(Xtrue, Xh_list[-1])))
    print("  RMSE: {}".format(sqrt(MSE(Xtrue, Xh_list[-1]))))
    print("  Mean SSI: {}".format(SSITensor(Xtrue, Xh_list[-1])))