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
directory = '../data/detections/'
# For Gabor filtered detections:
# directory = '../data/detections_Gabor/'

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
zg = -30

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
# -

# ## Run the full association process 

dt_kde = 0.5 # [s] Time resolution of the KDE (overlap)
bin_width = 1
dt_tol = int(0.5 * fs) # [samples] Tolerance for the time index when removing picks
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

# +
peaks = (npeakshf, npeakslf, speakshf, speakslf)
y_range_north = (n_selected_channels_m[1] - n_selected_channels_m[0])  # meters
y_range_south = (s_selected_channels_m[1] - s_selected_channels_m[0])  # meters
height_ratio = y_range_south / y_range_north

fig = dw.assoc.plot_associated_bicable_paper(peaks, n_longi_offset, pair_assoc, pair_loc, associations, localizations, n_cable_pos, s_cable_pos, n_dist, s_dist, dx, c0, fs, height_ratio)
fig.savefig('../figs/Figure6a.pdf', bbox_inches=None, transparent=True)
plt.show()

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

up_peaks = (n_up_peaks_hf, n_up_peaks_lf, s_up_peaks_hf, s_up_peaks_lf)
SNRs = (nSNRhf, nSNRlf, sSNRhf, sSNRlf)

peaks_close, snr_close = dw.assoc.apply_spatial_windows(up_peaks, SNRs, win_close)
peaks_mid, snr_mid = dw.assoc.apply_spatial_windows(up_peaks, SNRs, win_mid)
peaks_far, snr_far = dw.assoc.apply_spatial_windows(up_peaks, SNRs, win_far)

# +
iterations_far = 25
w_eval_far = 2
rms_threshold_far = 0.5

# Refine the cartesian grid and use only the points in the far window
# Create a grid of coordinates, choosing the spacing of the grid
dx_fargrid = 500 # [m]
dy_fargrid = 500 # [m]
xg_far, yg_far = np.meshgrid(np.arange(xf, x0, dx_fargrid), np.arange(y0, yf, dy_fargrid))

interpolator = RegularGridInterpolator((x, y),  bathy.T)
bathy_interp = interpolator((xg_far, yg_far))

# Remove points if the ocean depth is too shallow (i.e., less than -25 m)
mask = bathy_interp < -25
# Compute arrival times only for valid grid points
# Flatten the grid points
xg_far, yg_far = xg_far[mask], yg_far[mask]
# Filter grid points that are within the far window in terms of x-coordinate
xg_far_mask = (xg_far >= win_far[0]*dx) 
yg_far = yg_far[xg_far_mask]
xg_far = xg_far[xg_far_mask]

#Plot the two grids to check

plt.figure(figsize=(10, 8))
plt.scatter(xg, yg, c='blue', label='Initial Grid', alpha=0.5)
plt.scatter(xg_far, yg_far, c='red', label='Refined Grid', alpha=0.5)
plt.xlabel('X Coordinate (m)')
plt.ylabel('Y Coordinate (m)')
plt.title('Comparison of Initial and Refined Grids')
plt.legend()
# plt.axis('equal')
# plt.xlim(56000, 90000)
plt.grid()
plt.show()

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
