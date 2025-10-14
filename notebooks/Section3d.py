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
import pickle
plt.rcParams['font.size'] = 24
plt.rcParams['lines.linewidth'] = 3

# +
# Load the peak indexes and the metadata
directory = '../data/detections/'
# For Gabor filtered detections:
directory_gab = '../data/detections_Gabor/'

n_ds = xr.load_dataset(os.path.join(directory, 'peaks_indexes_tp_North_2021-11-04_08:06:42_ipi3_th_4.nc')) 
s_ds = xr.load_dataset(os.path.join(directory, 'peaks_indexes_tp_South_2021-11-04_08:06:42_ipi3_th_5.nc'))

n_ds_gab = xr.load_dataset(os.path.join(directory_gab, 'peaks_indexes_tp_North_2021-11-04_08:06:42_ipi3_th_4.nc')) 
s_ds_gab = xr.load_dataset(os.path.join(directory_gab, 'peaks_indexes_tp_South_2021-11-04_08:06:42_ipi3_th_5.nc'))

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
# Sort the peaks based on SNR difference
npeakshf, nSNRhf, npeakslf, nSNRlf = dw.detect.resolve_hf_lf_crosstalk(
    npeakshf, npeakslf, nSNRhf, nSNRlf, dt_tol=100, dx_tol=30
)

speakshf, sSNRhf, speakslf, sSNRlf = dw.detect.resolve_hf_lf_crosstalk(
    speakshf, speakslf, sSNRhf, sSNRlf, dt_tol=100, dx_tol=30
)

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
# Load associations
with open('../data/associations/Baseline_60iter/association_2021-11-04_08:06:42.pkl', 'rb') as f:
     baseline_assoc = pickle.load(f)

with open('../data/associations/FW_60iter/association_2021-11-04_08:06:42.pkl', 'rb') as f:
     fw_assoc = pickle.load(f)

with open('../data/associations/GaborFW_60iter/association_2021-11-04_08:06:42.pkl', 'rb') as f:
     gabor_assoc = pickle.load(f)


# +
nhf_assoc_list_pair = baseline_assoc['assoc_pair']['north']['hf'] # List to store paired associated picks for the North cable, HF calls
nlf_assoc_list_pair = baseline_assoc['assoc_pair']['north']['lf'] # List to store paired associated picks for the North cable, LF calls
nhf_assoc_list = baseline_assoc['assoc']['north']['hf'] # List to store associated picks for the North cable, HF calls
nlf_assoc_list = baseline_assoc['assoc']['north']['lf'] # List to store associated picks for the North cable, LF calls

shf_assoc_list_pair = baseline_assoc['assoc_pair']['south']['hf'] # List to store paired associated picks for the South cable, HF calls
slf_assoc_list_pair = baseline_assoc['assoc_pair']['south']['lf'] # List to store paired associated picks for the South cable, LF calls
shf_assoc_list = baseline_assoc['assoc']['south']['hf'] # List to store associated picks for the South cable, HF calls
slf_assoc_list = baseline_assoc['assoc']['south']['lf'] # List to store associated picks for the South cable, LF calls

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

# +
peaks = (npeakshf, npeakslf, speakshf, speakslf)
y_range_north = (n_selected_channels_m[1] - n_selected_channels_m[0])  # meters
y_range_south = (s_selected_channels_m[1] - s_selected_channels_m[0])  # meters
height_ratio = y_range_south / y_range_north

fig = dw.assoc.plot_associated_bicable_paper(peaks, n_longi_offset, pair_assoc, pair_loc, associations, localizations, n_cable_pos, s_cable_pos, n_dist, s_dist, dx, c0, fs, height_ratio)
fig.savefig('../figs/Figure6bis_a.pdf', bbox_inches=None, transparent=True)
plt.show()
# -

# # Far window association

# +
nhf_assoc_list_pair = fw_assoc['assoc_pair']['north']['hf'] # List to store paired associated picks for the North cable, HF calls
nlf_assoc_list_pair = fw_assoc['assoc_pair']['north']['lf'] # List to store paired associated picks for the North cable, LF calls
nhf_assoc_list = fw_assoc['assoc']['north']['hf'] # List to store associated picks for the North cable, HF calls
nlf_assoc_list = fw_assoc['assoc']['north']['lf'] # List to store associated picks for the North cable, LF calls

shf_assoc_list_pair = fw_assoc['assoc_pair']['south']['hf'] # List to store paired associated picks for the South cable, HF calls
slf_assoc_list_pair = fw_assoc['assoc_pair']['south']['lf'] # List to store paired associated picks for the South cable, LF calls
shf_assoc_list = fw_assoc['assoc']['south']['hf'] # List to store associated picks for the South cable, HF calls
slf_assoc_list = fw_assoc['assoc']['south']['lf'] # List to store associated picks for the South cable, LF calls

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
fig.savefig('../figs/Figure6bis_b.pdf', bbox_inches=None, transparent=True)
plt.show()

# # Gabor + far window association

# +
# load the peak indexes - North cable
npeakshf = n_ds_gab["peaks_indexes_tp_HF"].values  # Extract as NumPy array
npeakslf = n_ds_gab["peaks_indexes_tp_LF"].values
nSNRhf = n_ds_gab["SNR_hf"].values
nSNRlf = n_ds_gab["SNR_lf"].values

# load the peak indexes - South cable
speakshf = s_ds_gab["peaks_indexes_tp_HF"].values
speakslf = s_ds_gab["peaks_indexes_tp_LF"].values
sSNRhf = s_ds_gab["SNR_hf"].values
sSNRlf = s_ds_gab["SNR_lf"].values

# +
# Sort the peaks based on SNR difference
npeakshf, nSNRhf, npeakslf, nSNRlf = dw.detect.resolve_hf_lf_crosstalk(
    npeakshf, npeakslf, nSNRhf, nSNRlf, dt_tol=100, dx_tol=30
)

speakshf, sSNRhf, speakslf, sSNRlf = dw.detect.resolve_hf_lf_crosstalk(
    speakshf, speakslf, sSNRhf, sSNRlf, dt_tol=100, dx_tol=30
)

# +
nhf_assoc_list_pair = gabor_assoc['assoc_pair']['north']['hf'] # List to store paired associated picks for the North cable, HF calls
nlf_assoc_list_pair = gabor_assoc['assoc_pair']['north']['lf'] # List to store paired associated picks for the North cable, LF calls
nhf_assoc_list = gabor_assoc['assoc']['north']['hf'] # List to store associated picks for the North cable, HF calls
nlf_assoc_list = gabor_assoc['assoc']['north']['lf'] # List to store associated picks for the North cable, LF calls

shf_assoc_list_pair = gabor_assoc['assoc_pair']['south']['hf'] # List to store paired associated picks for the South cable, HF calls
slf_assoc_list_pair = gabor_assoc['assoc_pair']['south']['lf'] # List to store paired associated picks for the South cable, LF calls
shf_assoc_list = gabor_assoc['assoc']['south']['hf'] # List to store associated picks for the South cable, HF calls
slf_assoc_list = gabor_assoc['assoc']['south']['lf'] # List to store associated picks for the South cable, LF calls

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

# +
peaks = (npeakshf, npeakslf, speakshf, speakslf)

fig = dw.assoc.plot_associated_bicable_paper(peaks, n_longi_offset, pair_assoc, pair_loc, associations, localizations, n_cable_pos, s_cable_pos, n_dist, s_dist, dx, c0, fs, height_ratio)
fig.savefig('../figs/Figure6bis_c.pdf', bbox_inches=None, transparent=True)
plt.show()
