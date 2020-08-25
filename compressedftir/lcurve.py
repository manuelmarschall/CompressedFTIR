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
from compressedftir.utils import sum_sq_nnz

try:
    import matlab

    # This is actually `matlab._internal`, but matlab/__init__.py
    # mangles the path making it appear as `_internal`.
    # Importing it under a different name would be a bad idea.
    from _internal.mlarray_utils import _get_strides, _get_mlsize

    def _wrapper__init__(self, arr):
        assert arr.dtype == type(self)._numpy_type
        self._python_type = type(arr.dtype.type().item())
        self._is_complex = np.issubdtype(arr.dtype, np.complexfloating)
        self._size = _get_mlsize(arr.shape)
        self._strides = _get_strides(self._size)[:-1]
        self._start = 0

        if self._is_complex:
            self._real = arr.real.ravel(order='F')
            self._imag = arr.imag.ravel(order='F')
        else:
            self._data = arr.ravel(order='F')

    _wrappers = {}
    def _define_wrapper(matlab_type, numpy_type):
        t = type(matlab_type.__name__, (matlab_type,), dict(
            __init__=_wrapper__init__,
            _numpy_type=numpy_type
        ))
        # this tricks matlab into accepting our new type
        t.__module__ = matlab_type.__module__
        _wrappers[numpy_type] = t

    _define_wrapper(matlab.double, np.double)
    _define_wrapper(matlab.single, np.single)
    _define_wrapper(matlab.uint8, np.uint8)
    _define_wrapper(matlab.int8, np.int8)
    _define_wrapper(matlab.uint16, np.uint16)
    _define_wrapper(matlab.int16, np.int16)
    _define_wrapper(matlab.uint32, np.uint32)
    _define_wrapper(matlab.int32, np.int32)
    _define_wrapper(matlab.uint64, np.uint64)
    _define_wrapper(matlab.int64, np.int64)
    _define_wrapper(matlab.logical, np.bool_)

    def as_matlab(arr):
        try:
            cls = _wrappers[arr.dtype.type]
        except KeyError:
            raise TypeError("Unsupported data type")
        return cls(arr)
except Exception:
    pass

def lcurve_value_gmrf(Z0, U, V, lapU, lapV):
    """
    computes the argument and values of an lcurve result,
    having the proposed regularized low-rank model

    Arguments:
        Z0 {array like} -- sub-sampled data
        U {array like} -- first model component
        V {array like} -- second model component
        lapU {array like} -- regularizer for U
        lapV {array like} -- regularizer for V

    Returns:
        list -- [||Z0 - UV||_Omega / ||Z0||_Omega , ||lapU U|| + ||lapV V||]
    """
    nnz = np.nonzero(Z0)
    Xh = U.dot(V)
    dd = Xh - Z0
    # order="C" due to the kronecker structure of the regularization matrices
    return [sum_sq_nnz(nnz, dd)/sum_sq_nnz(nnz, Z0), U.reshape(-1, order="C").dot(lapU.dot(U.reshape(-1, order="C"))) + V.reshape(-1, order="C").dot(lapV.dot(V.reshape(-1, order="C")))]
    # return [sum_sq_nnz(nnz, dd)/sum_sq_nnz(nnz, Z0), U.reshape(-1, order="F").dot(lapU.dot(U.reshape(-1, order="F"))) + V.reshape(-1, order="F").dot(lapV.dot(V.reshape(-1, order="F")))]

def get_corner_node_matlab(lcurve, debug=False):
    """
    Computes the optimal argument of a given l-curve.
    Note: Only approximately and discrete. 
    TODO: More elaborate using interpolation/extrapolation and 
          analytical differentiation.
          Or implement https://arxiv.org/abs/1608.04571

    Arguments:      
        lcurve {list} -- l-curve [[x1, y1], [x2, y2], ...]

    Returns:
        int -- Optimal argument that maximizes curvature
    """
    lx, ly = np.zeros(len(lcurve)), np.zeros(len(lcurve))
    for lia in range(len(lcurve)):
        lx[lia] = lcurve[lia][0]
        ly[lia] = lcurve[lia][1]

    import matlab.engine
    eng = matlab.engine.start_matlab()
    eng.addpath(eng.genpath('../dfg_lowrank/simulation-200820/'))
    import io
    out = io.StringIO()
    err = io.StringIO()
    l_opt, info = eng.L_corner(as_matlab(lx), as_matlab(ly), nargout=2, stdout=out, stderr=err)
    # print(l_opt)
    # print(info)
    # print(out.getvalue())
    l_opt = int(l_opt)
    # k = curvature_lcurve(lx, ly)
    # l_opt = np.argmax(k)
    # print(l_opt)
    if False:
        import matplotlib
        matplotlib.use("Qt4Agg")
        import matplotlib.pyplot as plt
        plt.figure()
        plt.title("L-curve")
        plt.plot([np.log10(lcurve[lia][0]) for lia in range(len(lcurve))], [np.log10(lcurve[lia][1]) for lia in range(len(lcurve))], '-xb', label="L-curve")
        plt.plot(np.log10(lcurve[l_opt][0]), np.log10(lcurve[l_opt][1]), 'or', label="optimal value")
        plt.xlabel("log(|| Y - UV ||)")
        plt.ylabel("log(|| L_U U || + || L_V V ||)")
        plt.show()

    return l_opt

def get_corner_node_prune(lcurve):
    """
    Computes the optimal argument for a given L-curve.
    Implements the python version of the matlab corner method of
    % Per Christian Hansen and Toke Koldborg Jensen, DTU Compute, DTU;
    % Giuseppe Rodriguez, University of Cagliari, Italy; Sept. 2, 2011.
    motivated by the Reference: P. C. Hansen, T. K. Jensen and G. Rodriguez, 
    "An adaptive pruning algorithm for the discrete L-curve criterion," 
    J. Comp. Appl. Math., 198 (2007), 483-492. 

    Arguments:
        lcurve {list} -- L-curve [[x1, y1], [x2, y2], ...]

    Raises:
        ValueError: Input list contains not equal length lists
        ValueError: Invalid data given

    Returns:
        int -- index of lcurve list that is "optimal"
    """
    rho, eta = np.zeros(len(lcurve)), np.zeros(len(lcurve))
    for lia in range(len(lcurve)):
        rho[lia] = lcurve[lia][0]
        eta[lia] = lcurve[lia][1]

    if len(rho) != len(eta):
        raise ValueError("both arrays must have the same size")
    fin = np.isfinite(rho + eta)
    nzr = np.array([False]*len(rho))
    nzr[np.nonzero(rho*eta)[0]] = True
    keep = fin & nzr
    if len(keep) < 1:
        raise ValueError("To few accepted data found")
    if len(keep) < len(rho):
        print("I had to trim the data due to NaN/Inf or zero values")
    rho = rho[keep]
    eta = eta[keep]
    if np.any(rho[:-1] < rho[1:]) or np.any(eta[:-1] > eta[1:]):
        print("Warning: L-curve lacks monotonicity")
    nP = len(rho)                                   # number of points
    P = np.log10(np.array([rho, eta])).T            # Coordinates of the loglog L-curve
    V = P[1:, :] - P[:-1, :]                        # The vectors defined by these coord
    v = np.sqrt(np.sum(V**2, axis=1))               # length of the vectors
    # W = V/np.tile(v, (1, 2));                     # Normalized vectors.
    W = np.zeros(V.shape)
    W[:, 0] = V[:, 0]/v
    W[:, 1] = V[:, 1]/v
    clist = []                                      # list of condidates
    p = np.min([5, nP])                             # number of vectors in pruned L-curve
    convex = 0                                      # Are the pruned L-curves convex
    I = np.argsort(v)[::-1]                         # Sort lengths decending
    while p < (nP-1)*2:
        elmts = np.sort(I[:np.min([p, nP-1])])
        candidate = Angles(W[elmts, :], elmts)
        # print("candidate p={}, {}".format(p, candidate))
        if candidate > 0:
            convex = 1
        if candidate not in clist:
            clist.append(candidate)
        candidate = Global_Behaviour(P, W[elmts, :], elmts)
        if candidate not in clist:
            clist.append(candidate)
        p = p*2
        # print(clist)
    if 0 not in clist:
        clist.insert(0, 0)  
    clist = np.sort(clist)

    vz = np.argwhere(np.diff(P[clist, 1]) >= np.abs(np.diff(P[clist, 0])))
    if len(vz) > 1:
        if vz[0] == 0:
            vz = vz[1:] 
    elif len(vz) == 1:
        if vz[0] == 0:
            vz = []
    if vz == [] or len(vz) == 0:
        index = clist[-1]
    else:
        vects = np.array([P[clist[1:], 0] - P[clist[:-1], 0], P[clist[1:], 1] - P[clist[:-1], 1]]).T
        vects = np.dot(np.diag(1/np.sqrt(np.sum(vects**2, 1))), vects)
        delta = vects[:-1, 0] * vects[1:, 1]- vects[1:, 0] * vects[:-1, 1]
        vv = np.argwhere(delta[vz-1] <= 0)
        # print(vv)
        # print(vz)
        if vv == [] or len(vv) == 0:
            index = clist[vz[-1]]
        else:
            index = clist[vz[vv[0]]]
    return int(index)

def Angles(W, kv):
    delta = W[:-1, 0]*W[1:, 1] - W[1:, 0]*W[:-1, 1]
    # print("delta: \n {}".format(delta))
    mm = np.min(delta)
    kk = np.argmin(delta)
    # print("mm {}, kk {}, kv(kk)= {}".format(mm, kk, kv[kk]))
    if mm < 0 :                                     # is it really a corner
        index = kv[kk] + 1
    else:                                           # if there is no corner: 0
        index = 0
    return index        

def Global_Behaviour(P, vects, elmts):
    hwedge = np.abs(vects[:, 1])
    In = np.argsort(hwedge)
    count = 0
    ln = len(In)-1
    mn = In[0]
    mx = In[-1]
    while mn>=mx:
        mx = np.max([mx, In[ln-count]])
        count = count + 1
        mn = np.min([mn, In[count]])
    if count > 1:
        I = 0; J = 0
        for i in range(count):
            for j in range(ln, ln-count, -1):
                if In[i] < In[j]:
                    I = In[i]
                    J = In[j]
                    break
            if I > 0:
                break
    else:
        I = In[0]
        J = In[ln]
    
    x3 = P[elmts[J]+1, 0] + (P[elmts[I], 1] - P[elmts[J]+1, 1])/(P[elmts[J]+1, 1]-P[elmts[J],1])*(P[elmts[J]+1, 0]-P[elmts[J],0])
    origin = np.array([x3, P[elmts[I], 1]]).T
    dists = (origin[0] - P[:, 0])**2 + (origin[1]-P[:, 1])**2
    return np.argmin(dists)