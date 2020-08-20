import numpy as np
from scipy.sparse import load_npz
from compressedftir.utils import stop_at_exception
import os

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
            err_str = "File does not exists, or ends with .npz and is neither a scipy sparse matrix or in compressed numpy format."
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

    known_file_hints = ["afm-ir"]

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
    else:
        raise ValueError("Unknown format_hint: {}\n chose from {}".format(file_hint, known_file_hints))
    return retval