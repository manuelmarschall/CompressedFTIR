# License
 
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

Using this software in publications requires citing the following 

DOI: ??? 
'''

## References

This repository contains the python code that is used in the paper 
* [1] Compressed FTIR spectroscopy using low-rank matrix reconstruction (to appear).

## Motivation
Reducing measurement times and datasets by implementing reconstruction methods is a usual mathematical tool.
In this project we develop a regularized low-rank matrix recovery algorithm to account for smoothness, sparsity and low-rank properties of the given data.

## Installation 

To run the library one needs a $\geq$ python 3.6 installation with the python packages
* numpy
* scipy
* matplotlib

### Installation using pip

Install via the python package manager `pip` using

TODO

<!---
```
  pip install compressedftir
```

--> 

Guides to install python under Linux, Windows and Mac can be found here: https://realpython.com/installing-python/

### Quick guide for Python under Windows:

1. Download Python https://www.python.org/downloads/release/python-382/ (bottom of the page: "Windows x86-64 executable installer") 
2. Install Python using the installer and check "Add Python x.x to Path"
3. Run a terminal, e.g. CMD
4. Check the installation by typing

	```
	python
	```
   a command prompt should appear such as 

	```
	C:\Users\Marschall\Projects\simple_bayes>python
	Python 3.8.2 (tags/v3.8.2:7b3ab59, Feb 25 2020, 22:45:29) [MSC v.1916 32 bit (Intel)] on win32
	Type "help", "copyright", "credits" or "license" for more information.
	>>>
	```


5. Close the Python prompt using
	```
	exit()
	```
6. Install dependencies
	```
	python -m pip install numpy scipy matplotlib
	```

## Implementation details

Some features and important files are mentioned in the following

* `run_recon.py` is a start script and contains a dummy 2D example using an l-curve criterion for choosing the regularization parameter in a smoothed matrix reconstruction approach. 
* `compressedftir.datareader` implements a variety of data formats and can be adapted to your file format
* `compressedftir.reconstruction.lowrank` implements the code that is described in the paper [1]


## Contact

Please contact `manuel.marschall@ptb.de`.
