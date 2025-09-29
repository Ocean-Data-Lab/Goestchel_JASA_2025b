# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: venv
#     language: python
#     name: python3
# ---

# # Attempt at associating the faint calls using line detection

from IPython import get_ipython
ipython = get_ipython()
ipython.run_line_magic("reload_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# %reload_ext autoreload
# %autoreload 2
# Imports   
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import das4whales as dw
import pandas as pd
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
import matplotlib.cm as cm
import matplotlib.colors as colors
from tqdm import tqdm
import dask.array as da
# from dask import delayed
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity
from scipy.optimize import curve_fit
import scipy.signal as sp
import os
plt.rcParams['font.size'] = 24
plt.rcParams['lines.linewidth'] = 3

# +
# Load the peak indexes and the metadata
# directory = '../data/detections/'
# For Gabor filtered detections:
directory = '../data/detections_Gabor/'

n_ds = xr.load_dataset(os.path.join(directory, 'peaks_indexes_tp_North_2021-11-04_02:00:02_ipi3_th_4.nc')) 
s_ds = xr.load_dataset(os.path.join(directory, 'peaks_indexes_tp_South_2021-11-04_02:00:02_ipi3_th_5.nc'))

# +
# Constants from the metadata

fs = n_ds.attrs['fs']
dx = n_ds.attrs['dx']
nnx = n_ds.attrs['data_shape'][0]
snx = s_ds.attrs['data_shape'][0]
n_selected_channels_m = n_ds.attrs['selected_channels_m']
s_selected_channels_m = s_ds.attrs['selected_channels_m']
    
# Constants management
c0 = 1480
n_selected_channels = dw.data_handle.get_selected_channels(n_selected_channels_m, dx)
s_selected_channels = dw.data_handle.get_selected_channels(s_selected_channels_m, dx)
n_begin_chan = n_selected_channels[0]
n_end_chan = n_selected_channels[1]
n_longi_offset = n_selected_channels[0] // n_selected_channels[2]
s_begin_chan = s_selected_channels[0]
s_end_chan = s_selected_channels[1]
s_longi_offset = s_selected_channels[0] // s_selected_channels[2]
n_dist = (np.arange(nnx) * n_selected_channels[2] + n_selected_channels[0]) * dx
s_dist = (np.arange(snx) * s_selected_channels[2] + s_selected_channels[0]) * dx
dx = dx * n_selected_channels[2]

# +
# load the peak indexes - North cable
npeakshf = n_ds["peaks_indexes_tp_HF"].values  # Extract as NumPy array
npeakslf = n_ds["peaks_indexes_tp_LF"].values
nSNRhf = n_ds["SNR_hf"].values
nSNRlf = n_ds["SNR_lf"].values

# load the peak indexes - South cable
speakshf = s_ds["peaks_indexes_tp_HF"].values
speakslf = s_ds["peaks_indexes_tp_LF"].values
sSNRhf = s_ds["SNR_hf"].values
sSNRlf = s_ds["SNR_lf"].values

# +
# Sort the peaks based on SNR difference
npeakshf, nSNRhf, npeakslf, nSNRlf = dw.detect.resolve_hf_lf_crosstalk(
    npeakshf, npeakslf, nSNRhf, nSNRlf, dt_tol=100, dx_tol=30
)

speakshf, sSNRhf, speakslf, sSNRlf = dw.detect.resolve_hf_lf_crosstalk(
    speakshf, speakslf, sSNRhf, sSNRlf, dt_tol=100, dx_tol=30
)
# -

# ## Plot the map 

# +
# Import the cable location
df_north = pd.read_csv('../data/north_DAS_multicoord.csv')
df_south = pd.read_csv('../data/south_DAS_multicoord.csv')


# Extract the part of the dataframe used for the time picking process
idx_shift0 = int(n_begin_chan - df_north["chan_idx"].iloc[0]) # Shift between the cable locations (starting at the beach) and the channel locations
idx_shiftn = int(n_end_chan - df_north["chan_idx"].iloc[-1])

df_north_used = df_north.iloc[idx_shift0:idx_shiftn:n_selected_channels[2]][:nnx]

idx_shift0 = int(s_begin_chan - df_south["chan_idx"].iloc[0]) # Shift between the cable locations (starting at the beach) and the channel locations
idx_shiftn = int(s_end_chan - df_south["chan_idx"].iloc[-1])

df_south_used = df_south.iloc[idx_shift0:idx_shiftn:s_selected_channels[2]][:snx]

# Import the bathymetry data
bathy, xlon, ylat = dw.map.load_bathymetry('../data/GMRT_OOI_RCA_Cables.grd')
print(f'Origin of the corrdinates. Latitude = {ylat[0]}, Longitude = {xlon[-1]}')

utm_x0, utm_y0 = dw.map.latlon_to_utm(xlon[0], ylat[0])
utm_xf, utm_yf = dw.map.latlon_to_utm(xlon[-1], ylat[-1])

# Change the reference point to the last point
x0, y0 = utm_xf - utm_x0, utm_y0 - utm_y0
xf, yf = utm_xf - utm_xf, utm_yf - utm_y0
print(xf, yf)
# # Create vectors of coordinates
utm_x = np.linspace(utm_x0, utm_xf, len(xlon))
utm_y = np.linspace(utm_y0, utm_yf, len(ylat))
x = np.linspace(x0, xf, len(xlon))
y = np.linspace(y0, yf, len(ylat))


# +
# dw.map.plot_cables2D(df_north, df_south, bathy, xlon, ylat)

# +

# Cable geometry (make it correspond to x,y,z = cable_pos[:, 0], cable_pos[:, 1], cable_pos[:, 2])
n_cable_pos = np.zeros((len(df_north_used), 3))
s_cable_pos = np.zeros((len(df_south_used), 3))

n_cable_pos[:, 0] = df_north_used['x']
n_cable_pos[:, 1] = df_north_used['y']
n_cable_pos[:, 2] = df_north_used['depth']

s_cable_pos[:, 0] = df_south_used['x']
s_cable_pos[:, 1] = df_south_used['y']
s_cable_pos[:, 2] = df_south_used['depth']

# +
from scipy.interpolate import RegularGridInterpolator

# Create a grid of coordinates, choosing the spacing of the grid
dx_grid = 2000 # [m]
dy_grid = 2000 # [m]
xg, yg = np.meshgrid(np.arange(xf, x0, dx_grid), np.arange(y0, yf, dy_grid))

ti = 0
zg = -40

interpolator = RegularGridInterpolator((x, y),  bathy.T)
bathy_interp = interpolator((xg, yg))

# Remove points if the ocean depth is too shallow (i.e., less than -25 m)
mask = bathy_interp < -25
# Compute arrival times only for valid grid points
# Flatten the grid points
xg, yg = xg[mask], yg[mask]

# In case of a meshgrid object (non flattened), use the following code:
# xg[~mask] = np.nan
# yg[~mask] = np.nan

# +
# Define KDE computation as a delayed function
def compute_kde(delayed_picks, t_kde, bin_width, weights=None):
    """Computes the KDE of the delayed picks.

    Parameters
    ----------
    delayed_picks : array-like
        Delayed picks array.
    t_kde : array-like
        Time grid for the KDE.
    bin_width : float
        Bin width for the KDE.

    Returns
    -------
    array-like
        KDE density values.  
    
    """
    if weights is not None:
        # Use weighted KDE, Scipy's gaussian_kde is faster that sklearn's KernelDensity for weighted KDE
        kde = gaussian_kde(delayed_picks, bw_method=bin_width/np.std(delayed_picks), weights=weights)
        density = kde(t_kde)
        # kde = KernelDensity(kernel="epanechnikov", bandwidth=bin_width, algorithm='ball_tree')
        # kde.fit(delayed_picks[:, None], sample_weight=weights) # Reshape to (n_samples, 1)
        # log_dens = kde.score_samples(t_kde[:, np.newaxis]) # Evaluate on grid
        # density = np.exp(log_dens) # Convert log-density to normal density
    else:
        kde = KernelDensity(kernel="epanechnikov", bandwidth=bin_width, algorithm='ball_tree')
        kde.fit(delayed_picks[:, None]) # Reshape to (n_samples, 1)
        log_dens = kde.score_samples(t_kde[:, np.newaxis]) # Evaluate on grid
        density = np.exp(log_dens) # Convert log-density to normal density
    return density


def compute_selected_picks(peaks, hyperbola, dt_sel, fs):
    """Selects picks that are closest to the hyperbola within a given time window."""
    selected_picks = ([], [])
    for i, idx in enumerate(peaks[1]):
        dist_idx = peaks[0][i]
        pick_time = idx / fs

        if hyperbola[dist_idx] - dt_sel < pick_time < hyperbola[dist_idx] + dt_sel:
            if dist_idx in selected_picks[0]:
                existing_idx = selected_picks[0].index(dist_idx)
                if abs(hyperbola[dist_idx] - pick_time) < abs(hyperbola[dist_idx] - selected_picks[1][existing_idx] / fs):
                    selected_picks[1][existing_idx] = idx  # Replace with closer pick
            else:
                selected_picks[0].append(dist_idx)
                selected_picks[1].append(idx)
    
    return np.array(selected_picks[0]), np.array(selected_picks[1])


def compute_curvature(w_times, w_distances):
    """Computes curvature using second derivatives."""
    ddx = np.diff(w_times)
    ddy = np.diff(w_distances)
    ddx2 = np.diff(ddx)
    ddy2 = np.diff(ddy)
    curvature = np.abs(ddx2 * ddy[1:] - ddx[1:] * ddy2) / (ddx[1:]**2 + ddy[1:]**2)**(3/2)
    return np.mean(curvature)


# -

# Compute KDEs for all delayed picks
# TODO: KDE number of points proportional to the number of picks (y-axis)?
dt_kde = 0.5 # [s] Time resolution of the KDE (overlap)
bin_width = 1
n_shape_x = xg.shape[0]
s_shape_x = xg.shape[0]
dt_sel = 1.4 # [s] Selected time "distance" from the theoretical arrival time
w_eval = 5 # [s] Width of the evaluation window for curvature estimation
# Set the number of iterations for testing
iterations = 40

# +
# Initialize the max_kde variable to enter the loop
n_associated_list = []
n_used_hyperbolas = []
n_rejected_list = []
n_rejected_hyperbolas = []

s_associated_list = []
s_used_hyperbolas = []
s_rejected_list = []
s_rejected_hyperbolas = []

n_up_peaks_hf = np.copy(npeakshf)
s_up_peaks_hf = np.copy(speakshf)
n_up_peaks_lf = np.copy(npeakslf)
s_up_peaks_lf = np.copy(speakslf)
n_arr_tg = dw.loc.calc_arrival_times(ti, n_cable_pos, (xg, yg, zg), c0)
s_arr_tg = dw.loc.calc_arrival_times(ti, s_cable_pos, (xg, yg, zg), c0)

# n_arr_tg -= np.min(n_arr_tg, axis=1, keepdims=True)
# s_arr_tg -= np.min(s_arr_tg, axis=1, keepdims=True)

print(n_arr_tg.shape, nnx, n_cable_pos.shape)
print(s_arr_tg.shape, snx, s_cable_pos.shape)

# n_arr_tg = n_arr_tg[np.min(n_arr_tg, axis=1) > 20]

# +
n_idx_times_hf = np.array(n_up_peaks_hf[1]) / fs # Update with the remaining peaks
n_idx_times_lf = np.array(n_up_peaks_lf[1]) / fs # Update with the remaining peaks
s_idx_times_hf = np.array(s_up_peaks_hf[1]) / fs # Update with the remaining peaks
s_idx_times_lf = np.array(s_up_peaks_lf[1]) / fs # Update with the remaining peaks

# Make a delayed picks array for all the grid points
# Broadcast the time indices delayed by the theoretical arrival times for the grid points

n_delayed_picks_hf = n_idx_times_hf[None, :] - n_arr_tg[:, n_up_peaks_hf[0]]
n_delayed_picks_lf = n_idx_times_lf[None, :] - n_arr_tg[:, n_up_peaks_lf[0]]
s_delayed_picks_hf = s_idx_times_hf[None, :] - s_arr_tg[:, s_up_peaks_hf[0]]
s_delayed_picks_lf = s_idx_times_lf[None, :] - s_arr_tg[:, s_up_peaks_lf[0]]

global_min = min(np.min(n_delayed_picks_hf), np.min(n_delayed_picks_lf), np.min(s_delayed_picks_hf), np.min(s_delayed_picks_lf))
global_max = max(np.max(n_delayed_picks_hf), np.max(n_delayed_picks_lf), np.max(s_delayed_picks_hf), np.max(s_delayed_picks_lf))
Nkde=np.ceil((global_max - global_min) / dt_kde).astype(int) + 1
t_kde = np.linspace(global_min, global_max, Nkde)

print(Nkde)

# -

n_kde_hf = np.array(Parallel(n_jobs=-1)(
    delayed(dw.assoc.fast_kde_rect)(n_delayed_picks_hf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=nSNRhf) 
    for i in range(n_shape_x)
))
n_kde_lf = np.array(Parallel(n_jobs=-1)(
    delayed(dw.assoc.fast_kde_rect)(n_delayed_picks_lf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=nSNRlf)
    for i in range(n_shape_x)
))
s_kde_hf = np.array(Parallel(n_jobs=-1)(
    delayed(dw.assoc.fast_kde_rect)(s_delayed_picks_hf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=sSNRhf)
    for i in range(s_shape_x)
))
s_kde_lf = np.array(Parallel(n_jobs=-1)(
    delayed(dw.assoc.fast_kde_rect)(s_delayed_picks_lf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=sSNRlf)
    for i in range(s_shape_x)
))

print(n_kde_hf.shape, n_kde_lf.shape)
print(n_delayed_picks_hf.shape, n_delayed_picks_lf.shape)
print(s_kde_hf.shape, s_kde_lf.shape)
print(s_delayed_picks_hf.shape, s_delayed_picks_lf.shape)

hf_kde = n_kde_hf + s_kde_hf
lf_kde = n_kde_lf + s_kde_lf

# +
# Find the maximum for the 4 kde sets 

n_max_kde_hf = np.argmax(n_kde_hf)
nhf_imax, nhf_tmax = np.unravel_index(n_max_kde_hf, n_kde_hf.shape)

n_max_kde_lf = np.argmax(n_kde_lf)
nlf_imax, nlf_tmax = np.unravel_index(n_max_kde_lf, n_kde_lf.shape)

s_max_kde_hf = np.argmax(s_kde_hf)
shf_imax, shf_tmax = np.unravel_index(s_max_kde_hf, s_kde_hf.shape)

s_max_kde_lf = np.argmax(s_kde_lf)
slf_imax, slf_tmax = np.unravel_index(s_max_kde_lf, s_kde_lf.shape)

print(f'North HF max kde: {n_max_kde_hf}, max index: {nhf_imax}, max time: {nhf_tmax}')
print(f'North LF max kde: {n_max_kde_lf}, max index: {nlf_imax}, max time: {nlf_tmax}')
print(f'South HF max kde: {s_max_kde_hf}, max index: {shf_imax}, max time: {shf_tmax}')
print(f'South LF max kde: {s_max_kde_lf}, max index: {slf_imax}, max time: {slf_tmax}')

# Find the maximum for the 2 combined kde sets
hf_max_kde = np.argmax(hf_kde)
hf_imax, hf_tmax = np.unravel_index(hf_max_kde, hf_kde.shape)

lf_max_kde = np.argmax(lf_kde)
lf_imax, lf_tmax = np.unravel_index(lf_max_kde, lf_kde.shape)

print(f'Combined HF max kde: {hf_max_kde}, max index: {hf_imax}, max time: {hf_tmax}')
print(f'Combined LF max kde: {lf_max_kde}, max index: {lf_imax}, max time: {lf_tmax}')

# +
# Print the delayed time for the maximum KDE on north and south cables, lf
# Calculate height ratios based on y-range
y_range_north = (n_selected_channels_m[1] - n_selected_channels_m[0])  # meters
y_range_south = (s_selected_channels_m[1] - s_selected_channels_m[0])  # meters
height_ratio = y_range_south / y_range_north
print(height_ratio)

# Calculate the common x-range for all subplots
x_min = min(min(s_delayed_picks_lf[lf_imax, :]), min(n_delayed_picks_lf[lf_imax, :]))
x_max = max(max(s_delayed_picks_lf[lf_imax, :]), max(n_delayed_picks_lf[lf_imax, :]))

fig, axes = plt.subplots(4, 1,figsize=(10,16), constrained_layout=True, sharex=True, sharey=False, gridspec_kw={'height_ratios': [1, 0.3, height_ratio, 0.3]})
axes[0].set_title('North Cable')
axes[0].scatter(n_delayed_picks_hf[hf_imax, :], (n_longi_offset + npeakshf[0][:]) * dx * 1e-3, label='HF', c=nSNRhf, s=nSNRhf*0.8, cmap='plasma', rasterized=True)
axes[0].scatter(n_delayed_picks_lf[lf_imax, :], (n_longi_offset + npeakslf[0][:]) * dx * 1e-3, label='LF', c=nSNRlf, s=nSNRlf*0.8, cmap='viridis', rasterized=True)
axes[0].set_xlim(x_min, x_max)
axes[0].grid(linestyle='--', alpha=0.5)
axes[0].set_ylabel('Distance [km]')
axes[0].set_aspect('equal', adjustable='datalim')

axes[1].plot(t_kde, n_kde_hf[nlf_imax, :], color='tab:orange', lw=3, label='HF')
axes[1].plot(t_kde, n_kde_lf[nlf_imax, :], color='tab:green', lw=3, label='LF')
# plt.bar(lf_bin_edges[:-1], lf_hist, width=bin_width, alpha=0.5, label="Histogram", color='grey', edgecolor='black')
# plt.xlim(4, 8)
axes[1].set_ylim(0, max(np.max(n_kde_hf), np.max(n_kde_lf)) * 1.1)
axes[1].grid(linestyle='--', alpha=0.5)
axes[1].legend()
axes[1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
axes[1].set_ylabel('KDE [-]')
axes[1].legend(loc='lower left')

axes[2].set_title('South Cable')
axes[2].scatter(s_delayed_picks_hf[hf_imax, :], (s_longi_offset + speakshf[0][:]) * dx * 1e-3, label='HF', c=sSNRhf, s=sSNRhf*0.8, cmap='plasma', rasterized=True)
axes[2].scatter(s_delayed_picks_lf[lf_imax, :], (s_longi_offset + speakslf[0][:]) * dx * 1e-3, label='LF', c=sSNRlf, s=sSNRlf*0.8, cmap='viridis', rasterized=True)
axes[2].grid(linestyle='--', alpha=0.5)
axes[2].set_ylabel('Distance [km]')
axes[2].set_aspect('equal', adjustable='datalim')


axes[3].plot(t_kde, s_kde_hf[hf_imax, :], color='tab:orange', lw=3, label='HF')
axes[3].plot(t_kde, s_kde_lf[lf_imax, :], color='tab:green', lw=3, label='LF')
axes[3].set_ylim(0, max(np.max(s_kde_hf), np.max(s_kde_lf)) * 1.1)
axes[3].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
axes[3].set_ylabel('KDE [-]')
axes[3].set_xlabel('Delayed time [s]')
axes[3].legend(loc='lower left')

plt.grid(linestyle='--', alpha=0.5)
fig.savefig('../figs/Figure3.pdf', bbox_inches='tight', transparent=True, format='pdf')
plt.show()

# -

max_time_hf = t_kde[hf_tmax]
max_time_lf = t_kde[lf_tmax]

# +
# Plot the hyberbola on top of the picks 
# Create figure
fig, axes = plt.subplots(2, 1, figsize=(10, 16), sharex=True, sharey=False, constrained_layout=True, gridspec_kw={'height_ratios': [1, height_ratio]})

# First subplot
sc1 = axes[0].scatter(npeakshf[1][:] / fs, (n_selected_channels_m[0] + npeakshf[0][:] * dx) * 1e-3, 
                         c='grey',  s=nSNRhf, rasterized=True, alpha=0.7)
sc1 = axes[0].scatter(npeakslf[1][:] / fs, (n_selected_channels_m[0] + npeakslf[0][:] * dx) * 1e-3,
                         c='grey',  s=nSNRlf, rasterized=True, alpha=0.7)

axes[0].plot(max_time_hf + n_arr_tg[hf_imax, :], n_dist/1e3, ls='-', lw=3, color='tab:orange', label='HF')
# axes[0].plot(max_time_hf + n_arr_tg[hf_imax, :] + dt_sel, n_dist/1e3, ls='--', lw=3, color='k')
# axes[0].plot(max_time_hf + n_arr_tg[hf_imax, :] - dt_sel, n_dist/1e3, ls='--', lw=3, color='k')

axes[0].fill_betweenx(n_dist/1e3, max_time_hf + n_arr_tg[hf_imax, :] - dt_sel, max_time_hf + n_arr_tg[hf_imax, :] + dt_sel, color='tab:orange', alpha=0.3, \
                      edgecolor='tab:orange', linewidth=1)

axes[0].plot(max_time_lf + n_arr_tg[lf_imax, :], n_dist/1e3, ls='-', lw=3, color='tab:green', label='LF')
# axes[0].plot(max_time_lf + n_arr_tg[lf_imax, :] + dt_sel, n_dist/1e3, ls='--', lw=3, color='k')
# axes[0].plot(max_time_lf + n_arr_tg[lf_imax, :] - dt_sel, n_dist/1e3, ls='--', lw=3, color='k')

axes[0].fill_betweenx(n_dist/1e3, max_time_lf + n_arr_tg[lf_imax, :] - dt_sel, max_time_lf + n_arr_tg[lf_imax, :] + dt_sel, color='tab:green', alpha=0.3, \
                      edgecolor='tab:green', linewidth=1)

axes[0].set_title('North Cable');
axes[0].set_ylabel('Distance [km]')
axes[0].grid(linestyle='--', alpha=0.5)
axes[0].set_ylim(min(n_dist/1e3), max(n_dist/1e3))
axes[0].set_aspect('equal', adjustable='box')

# Second subplot
sc3 = axes[1].scatter(speakshf[1][:] / fs, (s_selected_channels_m[0] + speakshf[0][:] * dx) * 1e-3, 
                         c='grey',  s=sSNRhf, rasterized=True, alpha=0.7)
sc3 = axes[1].scatter(speakslf[1][:] / fs, (s_selected_channels_m[0] + speakslf[0][:] * dx) * 1e-3,
                            c='grey',  s=sSNRlf, rasterized=True, alpha=0.7)

axes[1].plot(max_time_hf + s_arr_tg[hf_imax, :], s_dist/1e3, ls='-', lw=3, color='tab:orange', label='HF')
# axes[1].plot(max_time_hf + s_arr_tg[hf_imax, :] + dt_sel, s_dist/1e3, ls='--', lw=3, color='k')
# axes[1].plot(max_time_hf + s_arr_tg[hf_imax, :] - dt_sel, s_dist/1e3, ls='--', lw=3, color='k')

axes[1].fill_betweenx(s_dist/1e3, max_time_hf + s_arr_tg[hf_imax, :] - dt_sel, max_time_hf + s_arr_tg[hf_imax, :] + dt_sel, color='tab:orange', alpha=0.5, \
                      edgecolor='tab:orange', linewidth=1)

axes[1].plot(max_time_lf + s_arr_tg[lf_imax, :], s_dist/1e3, ls='-', lw=3, color='tab:green', label='LF')
# axes[1].plot(max_time_lf + s_arr_tg[lf_imax, :] + dt_sel, s_dist/1e3, ls='--', lw=3, color='k')
# axes[1].plot(max_time_lf + s_arr_tg[lf_imax, :] - dt_sel, s_dist/1e3, ls='--', lw=3, color='k')

axes[1].fill_betweenx(s_dist/1e3, max_time_lf + s_arr_tg[lf_imax, :] - dt_sel, max_time_lf + s_arr_tg[lf_imax, :] + dt_sel, color='tab:green', alpha=0.5, \
                      edgecolor='tab:green', linewidth=1)


axes[1].set_title('South Cable')
axes[1].set_xlabel('Time [s]')
axes[1].set_ylabel('Distance [km]')
axes[1].grid(linestyle='--', alpha=0.5)
# set xlim to the same as the first subplot
axes[1].set_xlim(min(npeakshf[1][:] / fs), max(npeakshf[1][:] / fs))
axes[1].set_ylim(min(s_dist/1e3), max(s_dist/1e3))
axes[1].set_xticks(np.arange(0, max(speakshf[1][:] / fs)+10, 10))
axes[1].set_aspect('equal', adjustable='box')

for ax in axes:
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)

plt.savefig('../figs/Figure5.pdf', bbox_inches='tight', transparent=True, format='pdf')
plt.show()
# -

# ## Run the full association process 

dt_kde = 0.5 # [s] Time resolution of the KDE (overlap)
bin_width = 1
dt_tol = int(0.8 * fs) # [samples] Tolerance for the time index when removing picks
n_shape_x = xg.shape[0]
s_shape_x = xg.shape[0]
dt_sel = 1.4 # [s] Selected time "distance" from the theoretical arrival time
w_eval = 5 # [s] Width of the evaluation window for curvature estimation
rms_threshold = 0.5
# Set the number of iterations for testing
iterations = 50

# +
n_up_peaks_hf = np.copy(npeakshf)
s_up_peaks_hf = np.copy(speakshf)
n_up_peaks_lf = np.copy(npeakslf)
s_up_peaks_lf = np.copy(speakslf)

n_arr_tg = dw.loc.calc_arrival_times(ti, n_cable_pos, (xg, yg, zg), c0)
s_arr_tg = dw.loc.calc_arrival_times(ti, s_cable_pos, (xg, yg, zg), c0)

# +
nhf_assoc_list_pair = [] # List to store paired associated picks for the North cable, HF calls
nlf_assoc_list_pair = [] # List to store paired associated picks for the North cable, LF calls
nhf_assoc_list = [] # List to store associated picks for the North cable, HF calls
nlf_assoc_list = [] # List to store associated picks for the North cable, LF calls
n_used_hyperbolas = []
n_rejected_list = []
n_rejected_hyperbolas = []

shf_assoc_list_pair = [] # List to store paired associated picks for the South cable, HF calls
slf_assoc_list_pair = [] # List to store paired associated picks for the South cable, LF calls
shf_assoc_list = [] # List to store associated picks for the South cable, HF calls
slf_assoc_list = [] # List to store associated picks for the South cable, LF calls
s_used_hyperbolas = []
s_rejected_list = []
s_rejected_hyperbolas = []

association_lists = [
    nhf_assoc_list_pair, nlf_assoc_list_pair, shf_assoc_list_pair, slf_assoc_list_pair,
    nhf_assoc_list, shf_assoc_list, nlf_assoc_list, slf_assoc_list
    ]

hyperbolas = [n_used_hyperbolas, s_used_hyperbolas]

rejected_lists = [
    n_rejected_list, s_rejected_list, n_rejected_hyperbolas, s_rejected_hyperbolas
]

# +
pbar = tqdm(range(iterations), desc="Associated calls: 0")

for iteration in pbar:
    results = dw.assoc.process_iteration(
    # Peak data
    n_up_peaks_hf, n_up_peaks_lf, s_up_peaks_hf, s_up_peaks_lf,
    nSNRhf, nSNRlf, sSNRhf, sSNRlf,
    # Grid data
    n_arr_tg, s_arr_tg, n_shape_x, s_shape_x,
    # Cable positions
    n_cable_pos, s_cable_pos, n_longi_offset, s_longi_offset,
    # Association lists
    association_lists,
    # Hyperbolas
    hyperbolas,
    # Rejected lists
    rejected_lists,
    # Parameters
    fs, dt_kde, bin_width, dt_sel, w_eval, rms_threshold, c0, dx, dt_tol,
    # Iteration info
    iteration)

    if results is None:
        print(f"Stopped association at iteration {iteration}.")
        break  # Skip to the next iteration if no results are returned

    (n_up_peaks_hf, n_up_peaks_lf, s_up_peaks_hf, s_up_peaks_lf,
    nSNRhf, nSNRlf, sSNRhf, sSNRlf,
    n_arr_tg, s_arr_tg, n_shape_x, s_shape_x, 
    association_lists, rejected_lists, hyperbolas) = results

    total_associations = sum(len(lst) for lst in association_lists)
    pbar.set_description(f"Associated calls: {total_associations}")

# +
# Clean the associations
dw.assoc.clean_pairs(nhf_assoc_list_pair, shf_assoc_list_pair, shf_assoc_list)
dw.assoc.clean_pairs(nlf_assoc_list_pair, slf_assoc_list_pair, slf_assoc_list)
dw.assoc.clean_pairs(shf_assoc_list_pair, nhf_assoc_list_pair, nhf_assoc_list)
dw.assoc.clean_pairs(slf_assoc_list_pair, nlf_assoc_list_pair, nlf_assoc_list)

dw.assoc.clean_singles(nhf_assoc_list)
dw.assoc.clean_singles(nlf_assoc_list)
dw.assoc.clean_singles(shf_assoc_list)
dw.assoc.clean_singles(slf_assoc_list)

# +
# Localize using the selected picks
nhf_pair_loc = dw.loc.loc_from_picks(nhf_assoc_list_pair, n_cable_pos, c0, fs, return_uncertainty=False)
nlf_pair_loc = dw.loc.loc_from_picks(nlf_assoc_list_pair, n_cable_pos, c0, fs, return_uncertainty=False)
shf_pair_loc = dw.loc.loc_from_picks(shf_assoc_list_pair, s_cable_pos, c0, fs, return_uncertainty=False)
slf_pair_loc = dw.loc.loc_from_picks(slf_assoc_list_pair, s_cable_pos, c0, fs, return_uncertainty=False)

nhf_localizations = dw.loc.loc_from_picks(nhf_assoc_list, n_cable_pos, c0, fs, return_uncertainty=False)
nlf_localizations = dw.loc.loc_from_picks(nlf_assoc_list, n_cable_pos, c0, fs, return_uncertainty=False)
shf_localizations = dw.loc.loc_from_picks(shf_assoc_list, s_cable_pos, c0, fs, return_uncertainty=False)
slf_localizations = dw.loc.loc_from_picks(slf_assoc_list, s_cable_pos, c0, fs, return_uncertainty=False)

pair_assoc = (nhf_assoc_list_pair, nlf_assoc_list_pair, shf_assoc_list_pair, slf_assoc_list_pair)
pair_loc = (nhf_pair_loc, nlf_pair_loc, shf_pair_loc, slf_pair_loc)
associations = (nhf_assoc_list, nlf_assoc_list, shf_assoc_list, slf_assoc_list)
localizations = (nhf_localizations, nlf_localizations, shf_localizations, slf_localizations)
# -

peaks = (npeakshf, npeakslf, speakshf, speakslf)
fig = dw.assoc.plot_associated_bicable_paper(peaks, n_longi_offset, pair_assoc, pair_loc, associations, localizations, n_cable_pos, s_cable_pos, n_dist, s_dist, dx, c0, fs, height_ratio)
fig.savefig('../figs/Figure6a.pdf', bbox_inches=None, transparent=True)
plt.show()

up_peaks = (n_up_peaks_hf, n_up_peaks_lf, s_up_peaks_hf, s_up_peaks_lf)
SNRs = (nSNRhf, nSNRlf, sSNRhf, sSNRlf)
selected_channels_m = (n_selected_channels_m, s_selected_channels_m)

# +
# apply the spatial windows to the peaks

print(np.min(n_dist), np.max(n_dist))
win_close = [np.min(n_dist), 56000]
win_mid = [35000, np.max(n_dist)]
win_far = [56000, np.max(s_dist)]

# Convert windows to indexes
win_close = [int(win_close[0] / dx)-n_longi_offset, int(win_close[1] / dx)-n_longi_offset]
win_mid = [int(win_mid[0] / dx)-n_longi_offset, int(win_mid[1] / dx)-n_longi_offset]
win_far = [int(win_far[0] / dx)-n_longi_offset, int(win_far[1] / dx)-n_longi_offset]

peaks_close, snr_close = dw.assoc.apply_spatial_windows(up_peaks, SNRs, win_close)
peaks_mid, snr_mid = dw.assoc.apply_spatial_windows(up_peaks, SNRs, win_mid)
peaks_far, snr_far = dw.assoc.apply_spatial_windows(up_peaks, SNRs, win_far)
# fig=dw.assoc.plot_tpicks_resolved(peaks_far, snr_far, selected_channels_m, dx, fs)


# -

iterations_far = 25
w_eval_far = 2
rms_threshold_far = 0.5
# Reinitialize the delay from the cartesian grid 
n_arr_tg = dw.loc.calc_arrival_times(ti, n_cable_pos, (xg, yg, zg), c0)
s_arr_tg = dw.loc.calc_arrival_times(ti, s_cable_pos, (xg, yg, zg), c0)


# +
n_up_peaks_hf = np.copy(peaks_far[0])
s_up_peaks_hf = np.copy(peaks_far[2])
n_up_peaks_lf = np.copy(peaks_far[1])
s_up_peaks_lf = np.copy(peaks_far[3])

nSNRhf = np.copy(snr_far[0])
nSNRlf = np.copy(snr_far[1])
sSNRhf = np.copy(snr_far[2])
sSNRlf = np.copy(snr_far[3])

# +
pbar = tqdm(range(iterations_far), desc="Associated calls, far window: 0")

for iteration in pbar:
    results = dw.assoc.process_iteration(
    # Peak data
    n_up_peaks_hf, n_up_peaks_lf, s_up_peaks_hf, s_up_peaks_lf,
    nSNRhf, nSNRlf, sSNRhf, sSNRlf,
    # Grid data
    n_arr_tg, s_arr_tg, n_shape_x, s_shape_x,
    # Cable positions
    n_cable_pos, s_cable_pos, n_longi_offset, s_longi_offset,
    # Association lists
    association_lists,
    # Hyperbolas
    hyperbolas,
    # Rejected lists
    rejected_lists,
    # Parameters
    fs, dt_kde, bin_width, dt_sel, w_eval_far, rms_threshold_far, c0, dx, dt_tol,
    # Iteration info
    iteration)

    if results is None:
        print(f"Stopped association at iteration {iteration}.")
        break  # Exit the loop if no results are returned

    (n_up_peaks_hf, n_up_peaks_lf, s_up_peaks_hf, s_up_peaks_lf,
    nSNRhf, nSNRlf, sSNRhf, sSNRlf,
    n_arr_tg, s_arr_tg, n_shape_x, s_shape_x, 
    association_lists, rejected_lists, hyperbolas) = results

    total_associations = sum(len(lst) for lst in association_lists)
    pbar.set_description(f"Associated calls, far window: {total_associations}")

# +
dw.assoc.clean_pairs(nhf_assoc_list_pair, shf_assoc_list_pair, shf_assoc_list)
dw.assoc.clean_pairs(nlf_assoc_list_pair, slf_assoc_list_pair, slf_assoc_list)
dw.assoc.clean_pairs(shf_assoc_list_pair, nhf_assoc_list_pair, nhf_assoc_list)
dw.assoc.clean_pairs(slf_assoc_list_pair, nlf_assoc_list_pair, nlf_assoc_list)

dw.assoc.clean_singles(nhf_assoc_list)
dw.assoc.clean_singles(nlf_assoc_list)
dw.assoc.clean_singles(shf_assoc_list)
dw.assoc.clean_singles(slf_assoc_list)

# +
nhf_pair_loc = dw.loc.loc_from_picks(nhf_assoc_list_pair, n_cable_pos, c0, fs, return_uncertainty=False)
nlf_pair_loc = dw.loc.loc_from_picks(nlf_assoc_list_pair, n_cable_pos, c0, fs, return_uncertainty=False)
shf_pair_loc = dw.loc.loc_from_picks(shf_assoc_list_pair, s_cable_pos, c0, fs, return_uncertainty=False)
slf_pair_loc = dw.loc.loc_from_picks(slf_assoc_list_pair, s_cable_pos, c0, fs, return_uncertainty=False)

nhf_localizations = dw.loc.loc_from_picks(nhf_assoc_list, n_cable_pos, c0, fs, return_uncertainty=False)
nlf_localizations = dw.loc.loc_from_picks(nlf_assoc_list, n_cable_pos, c0, fs, return_uncertainty=False)
shf_localizations = dw.loc.loc_from_picks(shf_assoc_list, s_cable_pos, c0, fs, return_uncertainty=False)
slf_localizations = dw.loc.loc_from_picks(slf_assoc_list, s_cable_pos, c0, fs, return_uncertainty=False)

pair_assoc = (nhf_assoc_list_pair, nlf_assoc_list_pair, shf_assoc_list_pair, slf_assoc_list_pair)
pair_loc = (nhf_pair_loc, nlf_pair_loc, shf_pair_loc, slf_pair_loc)
associations = (nhf_assoc_list, nlf_assoc_list, shf_assoc_list, slf_assoc_list)
localizations = (nhf_localizations, nlf_localizations, shf_localizations, slf_localizations)
# -

fig = dw.assoc.plot_associated_bicable_paper(peaks, n_longi_offset, pair_assoc, pair_loc, associations, localizations, n_cable_pos, s_cable_pos, n_dist, s_dist, dx, c0, fs, height_ratio)
fig.savefig('../figs/Figure6b.pdf', bbox_inches=None, transparent=True)
plt.show()

s_rejected_list = rejected_lists[1]
dw.assoc.plot_reject_pick(speakshf, s_longi_offset, s_dist, dx, shf_assoc_list_pair, s_rejected_list, s_rejected_hyperbolas, fs)
plt.show()

up_peaks = (n_up_peaks_hf, n_up_peaks_lf, s_up_peaks_hf, s_up_peaks_lf)
SNRs = (nSNRhf, nSNRlf, sSNRhf, sSNRlf)
selected_channels_m = (n_selected_channels_m, s_selected_channels_m)
fig=dw.assoc.plot_tpicks_resolved(peaks_far, snr_far, selected_channels_m, dx, fs)

