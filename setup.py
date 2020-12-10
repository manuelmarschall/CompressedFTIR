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

import setuptools

with open("README.md", "r") as fh:

    long_description = fh.read()

setuptools.setup(

    name="compressedftir",  # Replace with your username

    version="0.0.2",

    author="Manuel Marschall",

    author_email="manuelmarschall@ptb.de",

    description="Reconstruction of sample FTIR data",

    long_description=long_description,

    long_description_content_type="text/markdown",

    url="https://github.com/manuelmarschall/CompressedFTIR",

    # packages=setuptools.find_packages(),
    packages=["compressedftir"],

    classifiers=[

        "Programming Language :: Python :: 3",

        "Operating System :: OS Independent",

    ],

    python_requires='>=3.6',

)
