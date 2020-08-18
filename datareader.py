import numpy as np
from scipy.sparse import load_npz

def stop_at_exception(ex, err_str=None):
    if err_str is not None:
        print(err_str)
    print(ex)
    print(" exit!")
    exit()

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

    # TODO: add csv AFM data
    assert isinstance(filepath, str)
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
        return retval
    