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
from scipy.sparse import load_npz
from scipy.io import loadmat
from compressedftir.utils import stop_at_exception
import os


def load_liquid_mat(filepath):
    """
    Loads a matlab .mat file.
    Assumes the .mat container has a field `tensor` and is given as a 3D container.
    Here, the dimensions of the data was 20x10x684 for all
    measurements conducted.

    Arguments:
        filepath {str} -- path to file

    Raises:
        KeyError: tensor key not in container

    Returns:
        array-like -- 3D data cube
    """

    tab = loadmat(filepath)
    assert isinstance(tab, dict)

    try:
        data = np.array(tab["tensor"])          # Data is already a tensor
    except KeyError:
        raise KeyError("tensor is not in struct, which is expected to be included")

    assert data.shape == (20, 10, 684)
    return data


def load_122020_mat(filepath):
    """
    Loads a matlab .mat file.
    Assumes the .mat container has a field `data1` and is given as a 3D container.
    Here, the dimensions of the data was 55x55x45 for all
    measurements conducted.
    Together with `data1` a 3D tensor of the same size is given, that indicates
    the size of the actual measurand. `rows_in1` is a 55x55x45 tensor of integers
    indicating the index of the measurement on the interferometer axis

    WARNING: The data is complex valued
    Arguments:
        filepath {str} -- path to file

    Raises:
        KeyError: data1 key not in container

    Returns:
        array-like -- 3D data cube
    """

    tab = loadmat(filepath)
    assert isinstance(tab, dict)

    n = 55
    try:
        data = np.array(tab["data1"])          # Data is already a tensor
    except KeyError:
        raise KeyError("data1 is not in struct, which is expected to be included")

    try:
        indices = np.array(tab["rows_in1"])          # Data is already a tensor
    except KeyError:
        raise KeyError("rows_in1 is not in struct, which is expected to be included")

    try:
        size_N = int(np.array(tab["N"]))          # Size of interferometer axis (integer)
    except KeyError:
        raise KeyError("N is not in struct, which is expected to be included")

    # the respective conversion (super-resolution) according to
    retval = np.zeros((n, n, size_N), dtype=np.complex)
    for lia in range(indices.shape[2]):
        retval[:, :, indices[0, 0, lia]] = data[:, :, lia]

    return retval


def load_FPA_mat(filepath):
    """
    Loads a matlab .mat file.
    Assumes the .mat container has a field "Expression1" and the
    3D data is column-wise raveled in the array.
    This method assumes data that was used for the low-rank Leishmania
    paper. Here, the dimensions of the data was 128x128x3554 for all
    measurements conducted.

    Arguments:
        filepath {str} -- path to file

    Raises:
        KeyError: Expression1 key not in container

    Returns:
        array-like -- 3D data cube
    """

    tab = loadmat(filepath)
    assert isinstance(tab, dict)

    n, m = 128, 3554
    try:
        data = np.array(tab["Expression1"])          # Data is expected to be flattened
    except KeyError:
        raise KeyError("Expression1 is not in struct, which is expected to be of type FPALeismania")
    data = data.reshape([n*n, m], order="C")
    data = data.reshape([n, n, m])
    return data


def load_afmir_csv(filepath):
    """
    Example for a conversion of AFMIR data files
    The positions x, y and t are read from file but not returned (TODO)

    Arguments:
        filepath {str} -- file path

    Returns:
        numpy array -- data matrix
    """
    t, x, y = [], [], []
    values = []
    with open(filepath, "r") as fp:
        lines = fp.readlines()
        for lia, line in enumerate(lines):
            line_arr = str.split(line, ",")
            if len(line_arr) < 3:
                # this row seems to be empty
                continue
            if lia == 0:
                # get amplitude values (in mV)
                # line_arr[0] and line_arr[1] should be empty in first row
                assert len(line_arr[0]) == 0 and len(line_arr[1]) == 0
                # remaining line contains the values
                for lib in range(2, len(line_arr)):
                    t.append(float(line_arr[lib]))
                continue
            loc_values = []
            x.append(float(line_arr[0]))
            y.append(float(line_arr[1]))
            for lib in range(2, len(line_arr)):
                loc_values.append(float(line_arr[lib]))
            values.append(loc_values)
    nx = len(np.unique(np.array(x)))
    ny = len(np.unique(np.array(y)))
    return np.fliplr(np.array(values).reshape([ny, nx, len(t)]).transpose([1, 0, 2]))


def load_npz_file(filepath):
    """
    Scipy sparse matrices or numpy compressed arrays are stored as .npz file

    Arguments:
        filepath {str} -- filename
    Returns:
        object -- Depends on stored data. Either scipy sparse matrix or numpy array
    """
    try:
        # Try scipy sparse format
        return load_npz(filepath)
    except IOError:
        # Nope. Is it compressed numpy?
        try:
            return np.load(filepath)
        except IOError as exception:
            # Nope neither
            err_str = "File does not exists, or ends with .npz and is neither a scipy "
            err_str += "sparse matrix or in compressed numpy format."
            stop_at_exception(exception, err_str)


def load_npy_file(filepath):
    """
    Numpy arrays are stored as .npy file

    Arguments:
        filepath {str} -- filename

    Returns:
        np.array -- Data array
    """
    try:
        # Try numpy format
        return np.load(filepath)
    except IOError as exception:
        stop_at_exception(exception, "File does not exists or cannot be read")
    except ValueError as exception:
        stop_at_exception(exception, "The file contains an object array, but allow_pickle=False given")


def load_data_file(filepath, format_hint=None):
    """
    Main loader function that distributes the loading according to filepath and format_hint

    Here you can implement your own file format. Just add a new format_hint and implement
    a data conversion in a new function.

    Arguments:
        filepath {str} -- path to file including file extension

    Keyword Arguments:
        format_hint {str} -- additional information which format is used (default: {None})

    Raises:
        ValueError: Unknown file type
        IOError   : File does not exist

    Returns:
        array-like -- sample data from file
    """
    # TODO: add csv AFM data
    if not isinstance(filepath, str):
        raise ValueError("filepath should be a string")
    if not os.path.exists(filepath):
        raise IOError("file does not exist: {}".format(filepath))

    known_file_hints = ["afm-ir",
                        "leishmania-fpa",
                        "122020",
                        "liquid"]

    if format_hint is None:
        # check for file suffix, as no hint is given
        if np.char.endswith(filepath, ".npz"):
            # scipy sparse format or numpy compressed
            retval = load_npz_file(filepath)
        elif np.char.endswith(filepath, ".npy"):
            # numpy matrix format
            retval = load_npy_file(filepath)
        else:
            raise ValueError("Unknown file suffix. exit!")
    elif format_hint == "afm-ir":
        if np.char.endswith(filepath, ".csv"):
            retval = load_afmir_csv(filepath)
        elif np.char.endswith(filepath, ".txt"):
            raise NotImplementedError(".txt conversion not implemented. Use .csv file.")
        else:
            raise ValueError("Unknown file suffix. exit!")
    elif format_hint == "leishmania-fpa":
        if not np.char.endswith(filepath, ".mat"):
            raise ValueError("unknown file suffix. Assuming .mat file")
        retval = load_FPA_mat(filepath)
    elif format_hint == "122020":
        if not np.char.endswith(filepath, ".mat"):
            raise ValueError("unknown file suffix. Assuming .mat file")
        retval = load_122020_mat(filepath)
    elif format_hint == "liquid":
        if not np.char.endswith(filepath, ".mat"):
            raise ValueError("unknown file suffix. Assuming .mat file")
        retval = load_liquid_mat(filepath)
    else:
        raise ValueError("Unknown format_hint: {}\n chose from {}".format(format_hint, known_file_hints))
    return retval
