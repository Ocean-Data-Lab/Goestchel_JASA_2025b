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
plt.rcParams['font.size'] = 24
plt.rcParams['lines.linewidth'] = 3

# Load the peak indexes and the metadata
n_ds = xr.load_dataset('../data/peaks_indexes_tp_North_2021-11-04_02:00:02_ipi3_th_4.nc') 
s_ds = xr.load_dataset('../data/peaks_indexes_tp_South_2021-11-04_02:00:02_ipi3_th_5.nc')

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
# replacing kde by histogram
hist_range = (t_kde[0], t_kde[-1])
bins = len(t_kde)
print(t_kde[2]-t_kde[1])
overlap = dt_kde
kernel_bins = int(np.round(bin_width / overlap))
print(kernel_bins)
if kernel_bins % 2 == 0:
    kernel_bins += 1  # Ensure odd length
kernel = np.ones(kernel_bins) / kernel_bins

hist, bin_edges = np.histogram(n_delayed_picks_hf[nhf_imax, :], bins=bins, range=hist_range, weights=nSNRhf)
hist = sp.convolve(hist, kernel, mode="same")
hist = hist / np.trapezoid(hist, t_kde)

s_hist, s_bin_edges = np.histogram(s_delayed_picks_hf[shf_imax, :], bins=bins, range=hist_range, weights=sSNRhf)
s_hist = sp.convolve(s_hist, kernel, mode="same")
s_hist = s_hist / np.trapezoid(s_hist, t_kde)

lf_hist, lf_bin_edges = np.histogram(n_delayed_picks_lf[nlf_imax, :], bins=bins, range=hist_range, weights=nSNRlf)
lf_hist = sp.convolve(lf_hist, kernel, mode="same")
lf_hist = lf_hist / np.trapezoid(lf_hist, t_kde)
s_lf_hist, s_lf_bin_edges = np.histogram(s_delayed_picks_lf[slf_imax, :], bins=bins, range=hist_range, weights=sSNRlf)
s_lf_hist = sp.convolve(s_lf_hist, kernel, mode="same")
s_lf_hist = s_lf_hist / np.trapezoid(s_lf_hist, t_kde)

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

dt_kde = 0.5 # [s] Time resolution of the KDE
bin_width = 1
# dt_kde = 0.25 # [s] Time resolution of the KDE (overlap)
# bin_width = 1.5
dt_tol = int(0.8 * fs) # [samples] Tolerance for the time index when removing picks
n_shape_x = xg.shape[0]
s_shape_x = xg.shape[0]
dt_sel = 1.4 # [s] Selected time "distance" from the theoretical arrival time
w_eval = 5 # [s] Width of the evaluation window for curvature estimation
rms_threshold = 0.5
# Set the number of iterations for testing
iterations = 5

# +
# Initialize the max_kde variable to enter the loop
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

n_up_peaks_hf = np.copy(npeakshf)
s_up_peaks_hf = np.copy(speakshf)
n_up_peaks_lf = np.copy(npeakslf)
s_up_peaks_lf = np.copy(speakslf)

n_arr_tg = dw.loc.calc_arrival_times(ti, n_cable_pos, (xg, yg, zg), c0)
s_arr_tg = dw.loc.calc_arrival_times(ti, s_cable_pos, (xg, yg, zg), c0)

# +
pbar = tqdm(range(iterations), desc="Associated calls: 0")

for iteration in pbar:
    # PART 1: PREPARE DATA AND COMPUTE KDEs
    # =====================================
    
    # Precompute the time indices from peaks for both frequency bands and cables
    n_idx_times_hf = np.array(n_up_peaks_hf[1]) / fs
    n_idx_times_lf = np.array(n_up_peaks_lf[1]) / fs
    s_idx_times_hf = np.array(s_up_peaks_hf[1]) / fs
    s_idx_times_lf = np.array(s_up_peaks_lf[1]) / fs

    # Calculate delayed picks for all grid points
    n_delayed_picks_hf = n_idx_times_hf[None, :] - n_arr_tg[:, n_up_peaks_hf[0]]
    n_delayed_picks_lf = n_idx_times_lf[None, :] - n_arr_tg[:, n_up_peaks_lf[0]]
    s_delayed_picks_hf = s_idx_times_hf[None, :] - s_arr_tg[:, s_up_peaks_hf[0]]
    s_delayed_picks_lf = s_idx_times_lf[None, :] - s_arr_tg[:, s_up_peaks_lf[0]]

    # Find the global min and max for KDE time range
    all_delayed_picks = [n_delayed_picks_hf, n_delayed_picks_lf, s_delayed_picks_hf, s_delayed_picks_lf]
    global_min = min(np.min(arr) for arr in all_delayed_picks)
    global_max = max(np.max(arr) for arr in all_delayed_picks)
    
    # Create time bins for KDE
    Nkde = np.ceil((global_max - global_min) / dt_kde).astype(int) + 1
    t_kde = np.linspace(global_min, global_max, Nkde)

    # Compute KDEs in parallel for each type
    # North high frequency
    n_kde_hf = np.array(Parallel(n_jobs=-1)(
        delayed(dw.assoc.fast_kde_rect)(n_delayed_picks_hf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=nSNRhf) 
        for i in range(n_shape_x)
    ))

    # North low frequency
    n_kde_lf = np.array(Parallel(n_jobs=-1)(
        delayed(dw.assoc.fast_kde_rect)(n_delayed_picks_lf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=nSNRlf)
        for i in range(n_shape_x)
    ))

    # South high frequency
    s_kde_hf = np.array(Parallel(n_jobs=-1)(
        delayed(dw.assoc.fast_kde_rect)(s_delayed_picks_hf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=sSNRhf)
        for i in range(s_shape_x)
    ))

    # South low frequency
    s_kde_lf = np.array(Parallel(n_jobs=-1)(
        delayed(dw.assoc.fast_kde_rect)(s_delayed_picks_lf[i, :], t_kde, overlap=dt_kde, bin_width=bin_width, weights=sSNRlf)
        for i in range(s_shape_x)
    ))

    # Reduced the number of grid points to speed up the process 
    if iteration == 0:  
        sum_kde = n_kde_hf + n_kde_lf + s_kde_hf + s_kde_lf
        maxsum = np.max(sum_kde, axis=1)
        binary = np.ones_like(maxsum)
        threshold = np.percentile(maxsum, 40)  # keep top 55%
        grid_mask = maxsum >= threshold
        n_arr_tg = n_arr_tg[grid_mask]
        s_arr_tg = s_arr_tg[grid_mask]
        n_shape_x = n_arr_tg.shape[0]
        s_shape_x = s_arr_tg.shape[0]

    # PART 2: FIND MAXIMA AND COMPUTE THEORETICAL ARRIVALS
    # ===================================================
    
    # Combine KDEs for high and low frequencies
    hf_kde = n_kde_hf + s_kde_hf  # Combined HF KDE from north and south
    lf_kde = n_kde_lf + s_kde_lf  # Combined LF KDE from north and south

    # Find maxima for HF KDE
    hf_max_idx = np.argmax(hf_kde)
    hf_imax, hf_tmax = np.unravel_index(hf_max_idx, hf_kde.shape)
    max_time_hf = t_kde[hf_tmax]

    # Get values from individual KDEs at that time
    # north_vals = n_kde_hf[:, hf_tmax]
    # south_vals = s_kde_hf[:, hf_tmax]
    # if np.max(north_vals) > np.max(south_vals):
    #     hf_imax = np.argmax(north_vals)
    # else:
    #     hf_imax = np.argmax(south_vals)

    # Find maxima for LF KDE
    lf_max_idx = np.argmax(lf_kde)
    lf_imax, lf_tmax = np.unravel_index(lf_max_idx, lf_kde.shape)
    max_time_lf = t_kde[lf_tmax]

    # Get values from individual KDEs at that time
    # north_vals = n_kde_lf[:, lf_tmax]
    # south_vals = s_kde_lf[:, lf_tmax]
    # if np.max(north_vals) > np.max(south_vals):
    #     lf_imax = np.argmax(north_vals)
    # else:
    #     lf_imax = np.argmax(south_vals)

    # Compute theoretical arrival times (hyperbolas)
    nhf_hyperbola = max_time_hf + n_arr_tg[hf_imax, :]  # North HF theoretical arrivals
    shf_hyperbola = max_time_hf + s_arr_tg[hf_imax, :]  # South HF theoretical arrivals
    nlf_hyperbola = max_time_lf + n_arr_tg[lf_imax, :]  # North LF theoretical arrivals
    slf_hyperbola = max_time_lf + s_arr_tg[lf_imax, :]  # South LF theoretical arrivals

    # PART 3: SELECT PICKS AND COMPUTE RESIDUALS
    # =========================================
    
    # Select picks around each hyperbola within +/- dt_sel
    nhf_idx_dist, nhf_idx_time = dw.assoc.select_picks(n_up_peaks_hf, nhf_hyperbola, dt_sel, fs)
    shf_idx_dist, shf_idx_time = dw.assoc.select_picks(s_up_peaks_hf, shf_hyperbola, dt_sel, fs)
    nlf_idx_dist, nlf_idx_time = dw.assoc.select_picks(n_up_peaks_lf, nlf_hyperbola, dt_sel, fs)
    slf_idx_dist, slf_idx_time = dw.assoc.select_picks(s_up_peaks_lf, slf_hyperbola, dt_sel, fs)

    # Calculate time indices
    nhf_times = nhf_idx_time / fs
    shf_times = shf_idx_time / fs
    nlf_times = nlf_idx_time / fs
    slf_times = slf_idx_time / fs

    # Define evaluation windows
    nhf_window_mask = dw.assoc.get_window_mask(nhf_times, w_eval)
    shf_window_mask = dw.assoc.get_window_mask(shf_times, w_eval)
    nlf_window_mask = dw.assoc.get_window_mask(nlf_times, w_eval)
    slf_window_mask = dw.assoc.get_window_mask(slf_times, w_eval)

    # Compute locations and residuals # TODO: try bicable localization here! 
    nhf_n, nhf_residuals = dw.assoc.loc_picks(nhf_idx_dist, nhf_idx_time, n_cable_pos, c0, fs)
    shf_n, shf_residuals = dw.assoc.loc_picks(shf_idx_dist, shf_idx_time, s_cable_pos, c0, fs)
    nlf_n, nlf_residuals = dw.assoc.loc_picks(nlf_idx_dist, nlf_idx_time, n_cable_pos, c0, fs)
    slf_n, slf_residuals = dw.assoc.loc_picks(slf_idx_dist, slf_idx_time, s_cable_pos, c0, fs)

    # Calculate RMS residuals
    nhf_rms = np.sqrt(np.mean(nhf_residuals[nhf_window_mask] ** 2))
    shf_rms = np.sqrt(np.mean(shf_residuals[shf_window_mask] ** 2))
    nlf_rms = np.sqrt(np.mean(nlf_residuals[nlf_window_mask] ** 2))
    slf_rms = np.sqrt(np.mean(slf_residuals[slf_window_mask] ** 2))

    # PART 4: ASSOCIATION LOGIC
    # ========================

    # Check all cases
    hf_north_south_good = nhf_rms < rms_threshold and shf_rms < rms_threshold
    only_hf_north_good = nhf_rms < rms_threshold and shf_rms >= rms_threshold
    only_hf_south_good = nhf_rms >= rms_threshold and shf_rms < rms_threshold
    
    lf_north_south_good = nlf_rms < rms_threshold and slf_rms < rms_threshold
    only_lf_north_good = nlf_rms < rms_threshold and slf_rms >= rms_threshold
    only_lf_south_good = nlf_rms >= rms_threshold and slf_rms < rms_threshold

    # HF and LF overlap
    if abs(max_time_hf - max_time_lf) < 1.4:
        if hf_kde[hf_imax, hf_tmax] > lf_kde[lf_imax, lf_tmax]:
            # HF is better
            lf_north_south_good = False
            only_lf_north_good = False
            only_lf_south_good = False
        else:
            # LF is better
            hf_north_south_good = False
            only_hf_north_good = False
            only_hf_south_good = False

    processed = False

    # Best case: Both HF and LF are good for both north and south
    # print(f"nhf_rms: {nhf_rms}, shf_rms: {shf_rms}, nlf_rms: {nlf_rms}, slf_rms: {slf_rms}")
    if hf_north_south_good and lf_north_south_good:
        # Process HF first (assuming it has priority)
        # North cable processing for HF
        #TODO: reselect picks using the new hyperbola ?
        #TODO: Fix the logic error happening when max_time_hf == max_time_lf, or filter picks? 

        if max_time_hf >= 0: # Do not associate the edge cases
            # snr = dw.assoc.select_snr(n_up_peaks_hf, nhf_idx_dist, nhf_idx_time, nSNRhf)
            mask_resi_n_hf = dw.assoc.filter_peaks(nhf_residuals, nhf_idx_dist, nhf_idx_time, n_longi_offset, dx)
            # mask_resi_n_hf = np.ones_like(nhf_residuals, dtype=bool)
            nhf_assoc_list_pair.append(np.asarray((nhf_idx_dist[mask_resi_n_hf], nhf_idx_time[mask_resi_n_hf])))
            n_used_hyperbolas.append(n_arr_tg[hf_imax, :])
            n_arr_tg[hf_imax, :] = dw.loc.calc_arrival_times(0, n_cable_pos, nhf_n[:3], c0)
            
            # South cable processing for HF
            mask_resi_s_hf = dw.assoc.filter_peaks(shf_residuals, shf_idx_dist, shf_idx_time, s_longi_offset, dx)
            # mask_resi_s_hf = np.ones_like(shf_residuals, dtype=bool)
            shf_assoc_list_pair.append(np.asarray((shf_idx_dist[mask_resi_s_hf], shf_idx_time[mask_resi_s_hf])))
            s_used_hyperbolas.append(s_arr_tg[hf_imax, :])
            s_arr_tg[hf_imax, :] = dw.loc.calc_arrival_times(0, s_cable_pos, shf_n[:3], c0)
        else:
            mask_resi_n_hf = np.one_like(nhf_residuals, dtype=bool)
            mask_resi_s_hf = np.one_like(shf_residuals, dtype=bool)
        
        # Then process LF
        if max_time_lf >= 0: # Do not associate the edge cases
            # North cable processing for LF
            mask_resi_n_lf = dw.assoc.filter_peaks(nlf_residuals, nlf_idx_dist, nlf_idx_time, n_longi_offset, dx)
            # mask_resi_n_lf = np.ones_like(nlf_residuals, dtype=bool)
            nlf_assoc_list_pair.append(np.asarray((nlf_idx_dist[mask_resi_n_lf], nlf_idx_time[mask_resi_n_lf])))
            n_used_hyperbolas.append(n_arr_tg[lf_imax, :])
            n_arr_tg[lf_imax, :] = dw.loc.calc_arrival_times(0, n_cable_pos, nlf_n[:3], c0)
            
            # South cable processing for LF
            mask_resi_s_lf = dw.assoc.filter_peaks(slf_residuals, slf_idx_dist, slf_idx_time, s_longi_offset, dx)
            # mask_resi_s_lf = np.ones_like(slf_residuals, dtype=bool)
            slf_assoc_list_pair.append(np.asarray((slf_idx_dist[mask_resi_s_lf], slf_idx_time[mask_resi_s_lf])))
            s_used_hyperbolas.append(s_arr_tg[lf_imax, :])
            s_arr_tg[lf_imax, :] = dw.loc.calc_arrival_times(0, s_cable_pos, slf_n[:3], c0)
        else:
            mask_resi_n_lf = np.ones_like(nlf_residuals, dtype=bool)
            mask_resi_s_lf = np.ones_like(slf_residuals, dtype=bool)
        
        # Remove all selected picks from both frequencies and both cables
        # Accurate indexes 
        n_up_peaks_hf, nSNRhf = dw.assoc.remove_peaks(n_up_peaks_hf, nhf_idx_dist, nhf_idx_time, mask_resi_n_hf, nSNRhf)
        n_up_peaks_lf, nSNRlf = dw.assoc.remove_peaks(n_up_peaks_lf, nlf_idx_dist, nlf_idx_time, mask_resi_n_lf, nSNRlf)
        s_up_peaks_hf, sSNRhf = dw.assoc.remove_peaks(s_up_peaks_hf, shf_idx_dist, shf_idx_time, mask_resi_s_hf, sSNRhf)
        s_up_peaks_lf, sSNRlf = dw.assoc.remove_peaks(s_up_peaks_lf, slf_idx_dist, slf_idx_time, mask_resi_s_lf, sSNRlf)

        # Fuzzy indexes (For peaks that are associated to hf or lf but also have points in the other band)
        n_up_peaks_hf, nSNRhf = dw.assoc.remove_peaks_tolerance(n_up_peaks_hf, nlf_idx_dist, nlf_idx_time, mask_resi_n_lf, nSNRhf, dt_tol=dt_tol)
        n_up_peaks_lf, nSNRlf = dw.assoc.remove_peaks_tolerance(n_up_peaks_lf, nhf_idx_dist, nhf_idx_time, mask_resi_n_hf, nSNRlf, dt_tol=dt_tol)
        s_up_peaks_hf, sSNRhf = dw.assoc.remove_peaks_tolerance(s_up_peaks_hf, slf_idx_dist, slf_idx_time, mask_resi_s_lf, sSNRhf, dt_tol=dt_tol)
        s_up_peaks_lf, sSNRlf = dw.assoc.remove_peaks_tolerance(s_up_peaks_lf, shf_idx_dist, shf_idx_time, mask_resi_s_hf, sSNRlf, dt_tol=dt_tol)

        processed = True

    # First priority: Case 1 - HF North and South are good
    elif hf_north_south_good:
        # North cable processing
        if max_time_hf >= 0: # Do not associate the edge cases
            mask_resi_n = dw.assoc.filter_peaks(nhf_residuals, nhf_idx_dist, nhf_idx_time, n_longi_offset, dx)
            # mask_resi_n = np.ones_like(nhf_residuals, dtype=bool)
            nhf_assoc_list_pair.append(np.asarray((nhf_idx_dist[mask_resi_n], nhf_idx_time[mask_resi_n])))
            n_used_hyperbolas.append(n_arr_tg[hf_imax, :])
            n_arr_tg[hf_imax, :] = dw.loc.calc_arrival_times(0, n_cable_pos, nhf_n[:3], c0)
        
            # South cable processing
            mask_resi_s = dw.assoc.filter_peaks(shf_residuals, shf_idx_dist, shf_idx_time, s_longi_offset, dx)
            # mask_resi_s = np.ones_like(shf_residuals, dtype=bool)
            shf_assoc_list_pair.append(np.asarray((shf_idx_dist[mask_resi_s], shf_idx_time[mask_resi_s])))
            s_used_hyperbolas.append(s_arr_tg[hf_imax, :])
            s_arr_tg[hf_imax, :] = dw.loc.calc_arrival_times(0, s_cable_pos, shf_n[:3], c0)
        else:
            mask_resi_n = np.ones_like(nhf_residuals, dtype=bool)
            mask_resi_s = np.ones_like(shf_residuals, dtype=bool)
        
        # Remove selected picks from both frequency bands (north)
        n_up_peaks_hf, nSNRhf = dw.assoc.remove_peaks(n_up_peaks_hf, nhf_idx_dist, nhf_idx_time, mask_resi_n, nSNRhf)
        n_up_peaks_lf, nSNRlf = dw.assoc.remove_peaks_tolerance(n_up_peaks_lf, nhf_idx_dist, nhf_idx_time, mask_resi_n, nSNRlf, dt_tol=dt_tol)

        # Remove selected picks from both frequency bands (south)
        s_up_peaks_hf, sSNRhf = dw.assoc.remove_peaks(s_up_peaks_hf, shf_idx_dist, shf_idx_time, mask_resi_s, sSNRhf)
        s_up_peaks_lf, sSNRlf = dw.assoc.remove_peaks_tolerance(s_up_peaks_lf, shf_idx_dist, shf_idx_time, mask_resi_s, sSNRlf, dt_tol=dt_tol)
        
        processed = True

    # Second priority: Case 2 - LF North and South are good  
    elif lf_north_south_good:
        # North cable processing
        if max_time_lf >= 0: # Do not associate the edge cases
            mask_resi_n = dw.assoc.filter_peaks(nlf_residuals, nlf_idx_dist, nlf_idx_time, n_longi_offset, dx)
            # mask_resi_n = np.ones_like(nlf_residuals, dtype=bool)
            nlf_assoc_list_pair.append(np.asarray((nlf_idx_dist[mask_resi_n], nlf_idx_time[mask_resi_n])))
            n_used_hyperbolas.append(n_arr_tg[lf_imax, :])
            n_arr_tg[lf_imax, :] = dw.loc.calc_arrival_times(0, n_cable_pos, nlf_n[:3], c0)

            # South cable processing
            mask_resi_s = dw.assoc.filter_peaks(slf_residuals, slf_idx_dist, slf_idx_time, s_longi_offset, dx)
            # mask_resi_s = np.ones_like(slf_residuals, dtype=bool)
            slf_assoc_list_pair.append(np.asarray((slf_idx_dist[mask_resi_s], slf_idx_time[mask_resi_s])))
            s_used_hyperbolas.append(s_arr_tg[lf_imax, :])
            s_arr_tg[lf_imax, :] = dw.loc.calc_arrival_times(0, s_cable_pos, slf_n[:3], c0)
        else:
            mask_resi_n = np.ones_like(nlf_residuals, dtype=bool)
            mask_resi_s = np.ones_like(slf_residuals, dtype=bool)

        # Remove selected picks from both frequency bands (north)
        n_up_peaks_lf, nSNRlf = dw.assoc.remove_peaks(n_up_peaks_lf, nlf_idx_dist, nlf_idx_time, mask_resi_n, nSNRlf)
        n_up_peaks_hf, nSNRhf = dw.assoc.remove_peaks_tolerance(n_up_peaks_hf, nlf_idx_dist, nlf_idx_time, mask_resi_n, nSNRhf, dt_tol=dt_tol)

        # Remove selected picks from both frequency bands (south)
        s_up_peaks_lf, sSNRlf = dw.assoc.remove_peaks(s_up_peaks_lf, slf_idx_dist, slf_idx_time, mask_resi_s, sSNRlf)
        s_up_peaks_hf, sSNRhf = dw.assoc.remove_peaks_tolerance(s_up_peaks_hf, slf_idx_dist, slf_idx_time, mask_resi_s, sSNRhf, dt_tol=dt_tol)
        
        processed = True
    
    # Lower priority cases - if neither combined case is good, try individual cables
    if not processed:
        # Case 3: Only HF North is good
        if only_hf_north_good:
            if max_time_hf >= 0:
                mask_resi = dw.assoc.filter_peaks(nhf_residuals, nhf_idx_dist, nhf_idx_time, n_longi_offset, dx)
                # mask_resi = np.ones_like(nhf_residuals, dtype=bool)
                nhf_assoc_list.append(np.asarray((nhf_idx_dist[mask_resi], nhf_idx_time[mask_resi])))
                n_used_hyperbolas.append(n_arr_tg[hf_imax, :])
                n_arr_tg[hf_imax, :] = dw.loc.calc_arrival_times(0, n_cable_pos, nhf_n[:3], c0)
            else:
                mask_resi = np.ones_like(nhf_residuals, dtype=bool)

            n_up_peaks_hf, nSNRhf = dw.assoc.remove_peaks(n_up_peaks_hf, nhf_idx_dist, nhf_idx_time, mask_resi, nSNRhf)
            n_up_peaks_lf, nSNRlf = dw.assoc.remove_peaks_tolerance(n_up_peaks_lf, nhf_idx_dist, nhf_idx_time, mask_resi, nSNRlf, dt_tol=dt_tol)
            processed = True
            
        # Case 4: Only HF South is good
        elif only_hf_south_good:
            if max_time_hf >= 0:
                mask_resi = dw.assoc.filter_peaks(shf_residuals, shf_idx_dist, shf_idx_time, s_longi_offset, dx)
                # mask_resi = np.ones_like(shf_residuals, dtype=bool)
                shf_assoc_list.append(np.asarray((shf_idx_dist[mask_resi], shf_idx_time[mask_resi])))
                s_used_hyperbolas.append(s_arr_tg[hf_imax, :])
                s_arr_tg[hf_imax, :] = dw.loc.calc_arrival_times(0, s_cable_pos, shf_n[:3], c0)
            else:
                mask_resi = np.ones_like(shf_residuals, dtype=bool)
            
            s_up_peaks_hf, sSNRhf = dw.assoc.remove_peaks(s_up_peaks_hf, shf_idx_dist, shf_idx_time, mask_resi, sSNRhf)
            s_up_peaks_lf, sSNRlf = dw.assoc.remove_peaks_tolerance(s_up_peaks_lf, shf_idx_dist, shf_idx_time, mask_resi, sSNRlf, dt_tol=dt_tol)
            processed = True
            
        # Case 5: Only LF North is good
        elif only_lf_north_good:
            if max_time_lf >= 0:
                mask_resi = dw.assoc.filter_peaks(nlf_residuals, nlf_idx_dist, nlf_idx_time, n_longi_offset, dx)
                # mask_resi = np.ones_like(nlf_residuals, dtype=bool)
                nlf_assoc_list.append(np.asarray((nlf_idx_dist[mask_resi], nlf_idx_time[mask_resi])))
                n_used_hyperbolas.append(n_arr_tg[lf_imax, :])
                n_arr_tg[lf_imax, :] = dw.loc.calc_arrival_times(0, n_cable_pos, nlf_n[:3], c0)
            else:
                mask_resi = np.ones_like(nlf_residuals, dtype=bool)
            
            n_up_peaks_lf, nSNRlf = dw.assoc.remove_peaks(n_up_peaks_lf, nlf_idx_dist, nlf_idx_time, mask_resi, nSNRlf)
            n_up_peaks_hf, nSNRhf = dw.assoc.remove_peaks_tolerance(n_up_peaks_hf, nlf_idx_dist, nlf_idx_time, mask_resi, nSNRhf, dt_tol=dt_tol)
            processed = True
            
        # Case 6: Only LF South is good
        elif only_lf_south_good:
            if max_time_lf >= 0:
                mask_resi = dw.assoc.filter_peaks(slf_residuals, slf_idx_dist, slf_idx_time, s_longi_offset, dx)
                # mask_resi = np.ones_like(slf_residuals, dtype=bool)
                slf_assoc_list.append(np.asarray((slf_idx_dist[mask_resi], slf_idx_time[mask_resi])))
                s_used_hyperbolas.append(s_arr_tg[lf_imax, :])
                s_arr_tg[lf_imax, :] = dw.loc.calc_arrival_times(0, s_cable_pos, slf_n[:3], c0)
            else:
                mask_resi = np.ones_like(slf_residuals, dtype=bool)
            
            s_up_peaks_lf, sSNRlf = dw.assoc.remove_peaks(s_up_peaks_lf, slf_idx_dist, slf_idx_time, mask_resi, sSNRlf)
            s_up_peaks_hf, sSNRhf = dw.assoc.remove_peaks_tolerance(s_up_peaks_hf, slf_idx_dist, slf_idx_time, mask_resi, sSNRhf, dt_tol=dt_tol)
            processed = True
    
    # Case 7: No good residuals - reject the hyperbolas
    if not processed:
        # Add the rejected hyperbolas to rejection lists
        n_rejected_list.append(np.asarray((nhf_idx_dist, nhf_idx_time)))
        n_rejected_list.append(np.asarray((nlf_idx_dist, nlf_idx_time)))
        n_rejected_hyperbolas.append(n_arr_tg[hf_imax, :])
        n_rejected_hyperbolas.append(n_arr_tg[lf_imax, :])
        
        s_rejected_list.append(np.asarray((shf_idx_dist, shf_idx_time)))
        s_rejected_list.append(np.asarray((slf_idx_dist, slf_idx_time)))
        s_rejected_hyperbolas.append(s_arr_tg[hf_imax, :])
        s_rejected_hyperbolas.append(s_arr_tg[lf_imax, :])
        
        # Remove the hyperbolas from the grid arrays
        n_arr_tg = np.delete(n_arr_tg, hf_imax, axis=0)
        s_arr_tg = np.delete(s_arr_tg, hf_imax, axis=0)
        n_shape_x = n_arr_tg.shape[0]
        s_shape_x = s_arr_tg.shape[0]

    # Update the progress bar with the number of associated calls
    association_lists = [
        nhf_assoc_list_pair, nlf_assoc_list_pair, shf_assoc_list_pair, slf_assoc_list_pair,
        nhf_assoc_list, shf_assoc_list, nlf_assoc_list, slf_assoc_list
        ]

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
nhf_pair_loc = dw.loc.loc_from_picks(nhf_assoc_list_pair, n_cable_pos, c0, fs)
nlf_pair_loc = dw.loc.loc_from_picks(nlf_assoc_list_pair, n_cable_pos, c0, fs)
shf_pair_loc = dw.loc.loc_from_picks(shf_assoc_list_pair, s_cable_pos, c0, fs)
slf_pair_loc = dw.loc.loc_from_picks(slf_assoc_list_pair, s_cable_pos, c0, fs)

nhf_localizations = dw.loc.loc_from_picks(nhf_assoc_list, n_cable_pos, c0, fs)
nlf_localizations = dw.loc.loc_from_picks(nlf_assoc_list, n_cable_pos, c0, fs)
shf_localizations = dw.loc.loc_from_picks(shf_assoc_list, s_cable_pos, c0, fs)
slf_localizations = dw.loc.loc_from_picks(slf_assoc_list, s_cable_pos, c0, fs)

pair_assoc = (nhf_assoc_list_pair, nlf_assoc_list_pair, shf_assoc_list_pair, slf_assoc_list_pair)
pair_loc = (nhf_pair_loc, nlf_pair_loc, shf_pair_loc, slf_pair_loc)
associations = (nhf_assoc_list, nlf_assoc_list, shf_assoc_list, slf_assoc_list)
localizations = (nhf_localizations, nlf_localizations, shf_localizations, slf_localizations)
# -

fig = dw.assoc.plot_associated_bicable_paper(npeakshf, speakslf, n_longi_offset, pair_assoc, pair_loc, associations, localizations, n_cable_pos, s_cable_pos, n_dist, s_dist, dx, c0, fs)
fig.savefig('../figs/Figure6.pdf', bbox_inches=None, transparent=True)
plt.show()


