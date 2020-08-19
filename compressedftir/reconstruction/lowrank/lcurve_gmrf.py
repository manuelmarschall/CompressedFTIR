from compressedftir.utils import (stop_at_exception, get_regularizer, get_nnz_indices, sum_sq_nnz, get_corner_node)
from compressedftir.reconstruction.lowrank.single_gmrf import lr_recon_single
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import multiprocessing as mp
import numpy as np
import os
import json
import time

def lcurve_value(Z0, U, V, lapU, lapV):
    nnz = np.nonzero(Z0)
    Xh = U.dot(V)
    dd = Xh - Z0
    return [sum_sq_nnz(nnz, dd)/sum_sq_nnz(nnz, Z0), U.reshape(-1, order="F").dot(lapU.dot(U.reshape(-1, order="F"))) + V.reshape(-1, order="F").dot(lapV.dot(V.reshape(-1, order="F")))]

def plot_lcurve(lcurve, l_opt, export_path):
    fig = plt.figure()
    plt.title("L-curve")
    plt.plot([np.log(lcurve[lia][0]) for lia in range(len(lcurve))], [np.log(lcurve[lia][1]) for lia in range(len(lcurve))], '-xb', label="L-curve")
    plt.plot(np.log(lcurve[l_opt][0]), np.log(lcurve[l_opt][1]), 'or', label="optimal value")
    plt.xlabel("log(|| Y - UV ||)")
    plt.ylabel("log(|| lapU U || + || lapV V ||)")
    fig.savefig(export_path + "lcurve.png")

def ex_fun(fun, queue, *args, **kwargs):
    try:
        fun(*args, **kwargs)
        queue.put({"success": True})
    except Exception as ex:
        queue.put({"success": False})
        stop_at_exception(ex, "Multiprocessing has terminated!")    
    
def wrap_mp(fun, *args, **kwargs):
    mpqueue = mp.Queue()
    p = mp.Process(target=ex_fun, args=(fun, mpqueue, *args), kwargs=kwargs)
    p.start()
    res = mpqueue.get()
    if not res["success"]:
        exit()
    p.join()
        
        
def plot_residual(res, export_path, label="", ylabel="", title=""):
    fig = plt.figure()
    plt.semilogy(res, '-r', label=label)
    plt.xlabel("iterations")
    plt.ylabel(ylabel)
    plt.title(title)
    if not label == "": 
        plt.legend()
    plt.tight_layout()
    fig.savefig(export_path)

def plot_results(Xh, Z0, export_path, title="", Xtrue=None):

    if Xtrue is not None:
        fig = plt.figure(figsize=(20, 5))    
        plt.subplot(141)
        plt.title("Full dataset")
        im = plt.imshow(Xtrue)
        plt.colorbar(im)
        plt.subplot(142)
        plt.title("Reconstruction")
        im = plt.imshow(Xh)
        plt.colorbar(im)
        plt.subplot(143)
        plt.title("Difference")
        im = plt.imshow(Xtrue - Xh)
        plt.colorbar(im)
        plt.subplot(144)
        plt.title("Sampleset")
        im = plt.imshow(Z0)
        plt.colorbar(im)
        plt.tight_layout()
    else:
        fig = plt.figure(figsize=(8, 4))    
        plt.subplot(121)
        plt.title("Reconstruction")
        im = plt.imshow(Xh)
        plt.colorbar(im)
        plt.subplot(122)
        plt.title("Sampleset")
        im = plt.imshow(Z0)
        plt.colorbar(im)
        plt.tight_layout()
    plt.suptitle(title)
    fig.subplots_adjust(top=0.88)
    fig.savefig(export_path)

def do_reconstruction(Z0, r, lam, tau=1e-2, max_iter=50, export_path=None, export_every_lambda_result=False, Xtrue=None):

    if export_every_lambda_result:
        if export_path is None:
            stop_at_exception(ValueError("You need to provide an export_path to allow exporting every step"))
    if len(Z0.shape) == 3:
        n, m, t = Z0.shape
        Z0 = Z0.reshape([-1, t])
        if Xtrue is not None:
            assert Xtrue.shape[0] == n
            assert Xtrue.shape[1] == m
            assert Xtrue.shape[2] == t
            Xtrue = Xtrue.reshape([-1, t])
        d3_flag = True
    elif len(Z0.shape) == 2:
        t = -1
        n, m = Z0.shape
        d3_flag = False
    else:
        raise ValueError("Shape of matrix not supported: {}".format(Z0.shape))

    U_list = []
    V_list = []
    res_l = []
    res_g = []
    if Xtrue is not None:
        res_true = []
    lcurve = dict()
    if not d3_flag:
        lapU, lapV = get_regularizer(np.sqrt(n), np.sqrt(n), Z0.shape[1], r)
    else:
        lapU, lapV = get_regularizer(n, m, Z0.shape[1], r)
    nnzU, nnzV = get_nnz_indices(Z0)
    solver_info = {
        "Xomega": Z0,
        "r": r,
        "T": max_iter,
        "tau": tau,
        "lapU": lapU,
        "lapV": lapV,
        "nnz_Z0_U": nnzU,
        "nnz_Z0_V": nnzV,
        "Xtrue": Xtrue
    }

    # start iteration over L-curve items
    glob_start_time = time.time()
    for lia, l in enumerate(lam):
        print("L-curve step: {}/{}".format(lia+1, len(lam)))
        exp_title = "Experimental setup:\n lambda: {}\n maximal rank: {}\n convergence tol: {}\n max iteration: {}".format(l, r, tau, max_iter)
        print(exp_title)
        solver_info["l_regu"] = l
        loc_start_time = time.time()
        curr_result = lr_recon_single(**solver_info)
        loc_duration = time.time() - loc_start_time
        U, V = curr_result["U"], curr_result["V"]
        res_l.append(curr_result["resL"])
        res_g.append(curr_result["resG"])
        lcurve[lia] = lcurve_value(Z0, U, V, lapU, lapV)
        U_list.append(U)
        V_list.append(V)
        
        if export_every_lambda_result:
            curr_path = export_path + "iter{}/".format(lia)
            export_dict = {
                "U": U.tolist(),
                "V": V.tolist(),
                "lambda": l,
                "lcurve": lcurve[lia],
                "resL": res_l[lia],
                "resG": res_g[lia],
                "starttime": loc_start_time,
                "duration": loc_duration
            }
            
            if not os.path.exists(curr_path):
                os.makedirs(curr_path)        
            with open(curr_path + "result.dat", "w") as fp:
                json.dump(export_dict, fp)
            plot_title = "Reconstruction; lambda: {:.1e}, rank: {}, tol: {:.1e}, max it: {}".format(l, r, tau, max_iter)
            treat_T = lambda T: T[:, int(t/2)].reshape([n, m]) if d3_flag else T
            wrap_mp(plot_results, treat_T(U.dot(V)), treat_T(Z0), curr_path + "result.png", plot_title, None if Xtrue is None else treat_T(Xtrue))
            wrap_mp(plot_residual, res_l[lia], curr_path + "local_res.png", label="local res", ylabel="||X_t - X_{t-1}|| / ||X_{t-1}||", title=plot_title)
            wrap_mp(plot_residual, res_g[lia], curr_path + "global_res.png", label="global res", ylabel="||Y - X_t||_Omega / ||Y||", title=plot_title)
            
            if Xtrue is not None:
                wrap_mp(plot_residual, curr_result["resT"], curr_path + "res2full.png", label="res2Full", ylabel="||Y - X_t|| / ||Y||", title=plot_title)
    if export_path is not None:
        l_opt = get_corner_node(lcurve)
        
        Xh = U_list[l_opt].dot(V_list[l_opt]) 
        export_dict = {
                "U": U_list[l_opt].tolist(),
                "V": V_list[l_opt].tolist(),
                "lambda": lam.tolist(),
                "l_opt": l_opt.tolist(),
                "lcurve": lcurve,
                "resL": res_l[l_opt],
                "resG": res_g[l_opt],
                "starttime": glob_start_time,
                "duration": time.time() - glob_start_time
            }
        if not os.path.exists(export_path):
            os.makedirs(export_path)        
        with open(export_path + "result.dat", "w") as fp:
            json.dump(export_dict, fp)
        wrap_mp(plot_lcurve, lcurve, l_opt, export_path + "lcurve.png")
        wrap_mp(plot_results, treat_T(Xh), treat_T(Z0), export_path + "result.png", plot_title, None if Xtrue is None else treat_T(Xtrue))
        wrap_mp(plot_residual, res_l[l_opt], export_path + "local_res.png", label="local res")
        warp_mp(plot_residual, res_g[l_opt],  export_path + "global_res.png", label="global res")