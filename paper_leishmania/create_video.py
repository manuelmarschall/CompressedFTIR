import numpy as np
from matplotlib import ticker
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import matplotlib.gridspec as grd
from alea.utils.progress.percentage import PercentageBar
from compressedftir.datareader import load_data_file
import json

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'small',
          'figure.figsize': (15, 5),
         'axes.labelsize': 'small',
         'axes.titlesize': 'small',
         'xtick.labelsize': 'small',
         'ytick.labelsize': 'small'}
pylab.rcParams.update(params)
""" reads all relevant information about the dataset, the approximation and the optimization
    to create a video to display the results, together with an image presenting the convergence.

Parameters
----------
dataset : str
    name of the dataset
dataset_path : str
    path to look for the original dataset
result_path : str
    path to the solver results and where to store the image and video
num_images : int or str, optional
    number of images to put into the video, usually use "all" to create a full video
    smaller numbers are usefull for debugging.
"""
dataset = "L. tarentolae film"
filepath = "testdata/Leishmania_Ltar/2020-01-FPALeish_1.mat"
print("Load dataset")
Xtrue = load_data_file(filepath, format_hint="leishmania-fpa")       # Get data from file. 
n, m = Xtrue.shape[0], Xtrue.shape[1]
Xtrue = np.reshape(Xtrue, (-1, Xtrue.shape[2]))  -1
# Xtrue, _ = subtract_offset(data, n, m)

print("read approximation result")
# Xgmrf = sio.loadmat("{}Xgmrf.mat".format(result_path))
# Xgmrf = Xgmrf["Xgmrf"]
path = "testdata/Leishmania_Ltar/samplerun_005p/iter7/"
    
with open(path + "result.dat", "r") as fp:
    res = json.load(fp)
U, V = np.array(res["U"]), np.array(res["V"])
Xgmrf = np.dot(U, V).reshape(*Xtrue.shape) 
movie_id = np.random.randint(0, 10**9)

# print("read approximation results")
# plot_dict = read_LR_GMRF_MAP_R_output("{}output.txt".format(result_path))
# plot_dict["figsize"] = 10
# plot_dict["format"] = "png"
# plot_dict["dpi"] = 100
# write_LR_GMRF_MAP_R_output_fig("{}out".format(result_path), **plot_dict)

vmin = np.min(Xtrue)
vmax = np.max(Xtrue)

zoom_range = 20
print("Start video capture")
FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Approximation for dataset {}'.format(dataset), artist='Matplotlib',
                comment='Sweep through the interferometer domain of the measured dataset')
writer = FFMpegWriter(fps=2, metadata=metadata)

dpi = 100
fig = plt.figure(figsize=(720/dpi, 480/dpi))
fig.subplots_adjust(top=0.85)

gs = grd.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[20, 20, 1], wspace=0.2, hspace=0.5)

plt.suptitle("Offset corrected L. tarentolae film with 5% of data for reconstruction")

ax = plt.subplot(gs[0, 0])
im_true = ax.imshow(np.zeros((n, n)),  vmin=0, vmax=0, cmap="jet",aspect="equal", origin="lower")
ax.set_title("Full dataset")
ax.set_xlabel("Focal position x (pixel id)")
ax.set_ylabel("Focal position y (pixel id)")

ax = plt.subplot(gs[0, 1])
im_approx = ax.imshow(np.zeros((n, n)), vmin=0, vmax=0, cmap="jet", aspect="equal", origin="lower")
ax.set_title("Reconstructed dataset")
ax.set_xlabel("Focal position x (pixel id)")
plt.setp(ax.get_yticklabels(), visible=False)

ax = plt.subplot(gs[0, 2])
cbar = plt.colorbar(im_true, cax=ax)


ax_t = plt.subplot(gs[1, 0])
ax_t.plot(Xtrue[2500, :])
ax_t.set_xlabel("Interferometer position t/Δt")
ax_t.set_ylabel("Relative intensity (a.u)")

ax = plt.subplot(gs[1, 1])
im_diff = ax.imshow(np.zeros((n, n)),  vmin=0, vmax=0, cmap="jet", aspect="equal", origin="lower")
ax.set_title("Difference")
ax.set_xlabel("Focal position x (pixel id)")
ax.set_ylabel("Focal position y (pixel id)")
plt.setp(ax.get_yticklabels(), visible=False)

ax = plt.subplot(gs[1, 2])
cbar_diff = plt.colorbar(im_diff, cax=ax)

t_range = np.arange(1600, 1800, 1)
print(t_range)
with writer.saving(fig, "{}visualization.mp4".format("testdata/Leishmania_Ltar/"), dpi=dpi):
    bar = PercentageBar(len(t_range), notebook=False)
    for idx in t_range:
        bar.next()
        im_true.set_data(Xtrue.reshape([n, n, -1], order="C")[:, :, idx].T)
        im_true.set_clim(vmin=np.min(Xtrue[:, idx]), vmax=np.max(Xtrue[:, idx]))
        im_approx.set_data(Xgmrf.reshape([n, n, -1], order="C")[:, :, idx].T)
        im_approx.set_clim(vmin=np.min(Xtrue[:, idx]), vmax=np.max(Xtrue[:, idx]))
        
        im_diff.set_data((Xtrue - Xgmrf).reshape([n, n, -1], order="C")[:, :, idx].T)
        im_diff.set_clim(vmin=np.min(Xtrue[:, idx] - Xgmrf[:, idx]), vmax=np.max(Xtrue[:, idx] - Xgmrf[:, idx]))

        cbardiff_ticks = np.linspace(np.min(Xtrue[:, idx] - Xgmrf[:, idx]), np.max(Xtrue[:, idx] - Xgmrf[:, idx]), num=11, endpoint=True)
        cbar_diff.ax.set_yticklabels(["{:1.2g}".format(i) for i in cbardiff_ticks])
        
        cbar_ticks = np.linspace(np.min(Xtrue[:, idx]), np.max(Xtrue[:, idx]), num=11, endpoint=True)
        # cbar.set_ticks(cbar_ticks) 
        cbar.ax.set_yticklabels(["{:1.2g}".format(i) for i in cbar_ticks])

        ax_t.clear()
        t_axis = np.arange(1600, 1800, 1)
        ax_t.plot(t_axis, Xtrue[2500, t_axis])
        ax_t.set_title("Interferometer position")
        ax_t.axvline(idx, -1, 1, linestyle="--", color="red", label="t={}Δt".format(idx))
        ax_t.legend()
        ax_t.set_xlabel("Interferometer position t/Δt")
        ax_t.set_ylabel("Relative intensity (a.u)")

        writer.grab_frame()

fig.clear()
print("\nDone")