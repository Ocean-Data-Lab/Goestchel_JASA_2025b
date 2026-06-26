# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: 'defaultInterpreterPath: 3.13.5.final.0'
#     language: python
#     name: python3
# ---

# # Section III.D. code

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
plt.rcParams['font.size'] = 34
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
# -

peaks = (npeakshf, npeakslf, speakshf, speakslf)
selected_channels_m = (n_selected_channels_m, s_selected_channels_m)
y_range_north = (n_selected_channels_m[1] - n_selected_channels_m[0])  # meters
y_range_south = (s_selected_channels_m[1] - s_selected_channels_m[0])  # meters
height_ratio = y_range_south / y_range_north


# +
batch = '4'
timestamp = '2021-11-04_08:06:42'
annot_root = '../data/annotations/'


def get_time_dist(pairs, fs, dx, selected_channels, selected_channel_m):
    times = pairs[1] / fs
    dists = selected_channel_m[0] + pairs[0] * dx * selected_channels[2]
    return np.vstack([times, dists]).T

n_annot = pd.read_csv(f'{annot_root}/Batch{batch}/annotated_calls_north_{timestamp}.csv')
s_annot = pd.read_csv(f'{annot_root}/Batch{batch}/annotated_calls_south_{timestamp}.csv')

# +
# Plot annotations similar to plot_associated_bicable_paper
fig, axes = plt.subplots(2, 1, figsize=(10, 16), sharex=True, sharey=False, 
                         constrained_layout=True, 
                         gridspec_kw={'height_ratios': [1, height_ratio]})

# Separate HF and LF annotations for North cable
n_annot_hf = n_annot[n_annot['call_type'] == 'HF']
n_annot_lf = n_annot[n_annot['call_type'] == 'LF']

# Separate HF and LF annotations for South cable
s_annot_hf = s_annot[s_annot['call_type'] == 'HF']
s_annot_lf = s_annot[s_annot['call_type'] == 'LF']

# Get color palettes - matching plot_associated_bicable_paper
hf_palette = plt.get_cmap('YlOrRd_r')
lf_palette = plt.get_cmap('YlGnBu_r')

# Get unique call IDs
n_hf_call_ids = sorted(n_annot_hf['call_id'].unique())
n_lf_call_ids = sorted(n_annot_lf['call_id'].unique())
s_hf_call_ids = sorted(s_annot_hf['call_id'].unique())
s_lf_call_ids = sorted(s_annot_lf['call_id'].unique())

# Calculate total number of events for consistent color scaling
nbhf = max(len(n_hf_call_ids), len(s_hf_call_ids))
nblf = max(len(n_lf_call_ids), len(s_lf_call_ids))

# Generate colors with same convention as the paper function
start, end = 0.0, 0.6  # Avoids part of the colormap that is too light
hf_colors = [hf_palette(start + (end - start) * i / max(nbhf - 1, 1)) for i in range(nbhf)]
lf_colors = [lf_palette(start + (end - start) * i / max(nblf - 1, 1)) for i in range(nblf)]

# Plot North cable
ax_north = axes[0]

# Plot raw picks
ax_north.scatter(npeakshf[1][:] / fs, (n_longi_offset + npeakshf[0][:]) * dx * 1e-3,
                    label='All peaks', s=0.5, alpha=0.2, color='tab:gray', rasterized=True)
ax_north.scatter(npeakslf[1][:] / fs, (n_longi_offset + npeakslf[0][:]) * dx * 1e-3,
                    label='All peaks', s=0.5, alpha=0.2, color='tab:gray', rasterized=True)

# Plot HF annotations
for i, call_id in enumerate(n_hf_call_ids):
    call_data = n_annot_hf[n_annot_hf['call_id'] == call_id]
    ax_north.scatter(call_data['time'], call_data['dist'] * 1e-3, 
                    color=hf_colors[i], s=50, alpha=1.0, 
                    marker='o', rasterized=True)

# Plot LF annotations
for i, call_id in enumerate(n_lf_call_ids):
    call_data = n_annot_lf[n_annot_lf['call_id'] == call_id]
    ax_north.scatter(call_data['time'], call_data['dist'] * 1e-3, 
                    color=lf_colors[i], s=50, alpha=1.0, 
                    marker='o', rasterized=True)

ax_north.set_ylabel(r'$\mathbf{North\ Cable}$' + '\nDistance [km]')
ax_north.set_title('Annotated calls')
ax_north.set_ylim(n_selected_channels_m[0] * 1e-3, n_selected_channels_m[1] * 1e-3)
ax_north.set_xlim(0, 70)
ax_north.grid(linestyle='--', alpha=0.3, linewidth=0.5)
ax_north.set_axisbelow(True)
ax_north.spines[['top', 'right']].set_visible(False)

# Plot South cable
ax_south = axes[1]

# Plot raw picks
ax_south.scatter(speakshf[1][:] / fs, (s_longi_offset + speakshf[0][:]) * dx * 1e-3,
                    label='All peaks', s=0.5, alpha=0.2, color='tab:gray', rasterized=True)
ax_south.scatter(speakslf[1][:] / fs, (s_longi_offset + speakslf[0][:]) * dx * 1e-3,
                    label='All peaks', s=0.5, alpha=0.2, color='tab:gray', rasterized=True)

# Plot HF annotations
for i, call_id in enumerate(s_hf_call_ids):
    call_data = s_annot_hf[s_annot_hf['call_id'] == call_id]
    ax_south.scatter(call_data['time'], call_data['dist'] * 1e-3, 
                    color=hf_colors[i], s=50, alpha=1.0, 
                    marker='o', rasterized=True)

# Plot LF annotations
for i, call_id in enumerate(s_lf_call_ids):
    call_data = s_annot_lf[s_annot_lf['call_id'] == call_id]
    ax_south.scatter(call_data['time'], call_data['dist'] * 1e-3, 
                    color=lf_colors[i], s=50, alpha=1.0, 
                    marker='o', rasterized=True)

ax_south.set_xlabel('Time [s]')
ax_south.set_ylabel(r'$\mathbf{South\ Cable}$' + '\nDistance [km]')
ax_south.set_title('Annotated Calls')
ax_south.set_ylim(s_selected_channels_m[0] * 1e-3, s_selected_channels_m[1] * 1e-3)
ax_south.grid(linestyle='--', alpha=0.3, linewidth=0.5)
ax_south.set_axisbelow(True)
ax_south.spines[['top', 'right']].set_visible(False)

# Add a common legend matching the paper function style
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches

# Create a parent container for the legend with frame
legend_container = inset_axes(ax_south, width="25%", height="12%", loc='lower center',
                            bbox_to_anchor=(-0.5, -0.01, 2, 0.6), bbox_transform=ax_south.transAxes)

legend_container.set_xlim(0, 1)
legend_container.set_ylim(0, 1)
legend_container.set_xticks([])
legend_container.set_yticks([])
legend_container.set_facecolor('white')

# Make the background white with rounded box
rounded_box = patches.FancyBboxPatch((0, 0), 1, 1, 
                            boxstyle="round,pad=0.1", 
                            facecolor='white', 
                            edgecolor='gray', 
                            linewidth=1,
                            transform=legend_container.transAxes)
legend_container.add_patch(rounded_box)

# Calculate spacing and number of markers
max_markers_per_type = 9
marker_spacing = 0.4 / max_markers_per_type if max_markers_per_type > 1 else 0.2

# HF section (left half)
hf_start_x = 0.08
hf_title_x = hf_start_x + (max_markers_per_type - 1) * marker_spacing / 2
legend_container.text(hf_title_x, 0.6, 'HF calls', ha='center', va='center')

# Plot HF markers horizontally
n_hf_to_show = min(len(hf_colors), max_markers_per_type)
if len(hf_colors) <= max_markers_per_type:
    for i in range(n_hf_to_show):
        x_pos = hf_start_x + i * marker_spacing
        legend_container.scatter(x_pos, 0.2, color=hf_colors[i],
                            marker='o', s=100, alpha=1)
else:
    markers_each_side = (max_markers_per_type - 1) // 2
    for i in range(markers_each_side):
        x_pos = hf_start_x + i * marker_spacing
        legend_container.scatter(x_pos, 0.2, color=hf_colors[i],
                            marker='o', s=100, alpha=1)
    middle_x = hf_start_x + markers_each_side * marker_spacing
    legend_container.text(middle_x, 0.2, '...', ha='center', va='center')
    for i in range(markers_each_side):
        x_pos = hf_start_x + (markers_each_side + 1 + i) * marker_spacing
        color_idx = len(hf_colors) - markers_each_side + i
        legend_container.scatter(x_pos, 0.2, color=hf_colors[color_idx],
                            marker='o', s=100, alpha=1)

# LF section (right half)
lf_start_x = 0.58
lf_title_x = lf_start_x + (max_markers_per_type - 1) * marker_spacing / 2
legend_container.text(lf_title_x, 0.6, 'LF calls', ha='center', va='center')

# Plot LF markers horizontally
n_lf_to_show = min(len(lf_colors), max_markers_per_type)
if len(lf_colors) <= max_markers_per_type:
    for i in range(n_lf_to_show):
        x_pos = lf_start_x + i * marker_spacing
        legend_container.scatter(x_pos, 0.2, color=lf_colors[i],
                            marker='o', s=100, alpha=1)
else:
    markers_each_side = (max_markers_per_type - 1) // 2
    for i in range(markers_each_side):
        x_pos = lf_start_x + i * marker_spacing
        legend_container.scatter(x_pos, 0.2, color=lf_colors[i],
                            marker='o', s=100, alpha=1)
    middle_x = lf_start_x + markers_each_side * marker_spacing
    legend_container.text(middle_x, 0.2, '...', ha='center', va='center')
    for i in range(markers_each_side):
        x_pos = lf_start_x + (markers_each_side + 1 + i) * marker_spacing
        color_idx = len(lf_colors) - markers_each_side + i
        legend_container.scatter(x_pos, 0.2, color=lf_colors[color_idx],
                            marker='o', s=100, alpha=1)

# Vertical separator line between HF and LF
separator_x = 0.5
legend_container.axvline(x=separator_x, ymin=0.2, ymax=0.8, 
                        color='lightgray', linewidth=1, alpha=0.7)

plt.savefig(f'../figs/Figure6bis_a.pdf',  bbox_inches=None, transparent=True)
plt.show()

# Print summary statistics
print(f"\nAnnotation Summary:")
print(f"North Cable: {len(n_annot_hf)} HF picks, {len(n_annot_lf)} LF picks")
print(f"South Cable: {len(s_annot_hf)} HF picks, {len(s_annot_lf)} LF picks")
print(f"Total HF calls: {nbhf}, Total LF calls: {nblf}")
print(f"North Cable calls (HF): {n_hf_call_ids}")
print(f"North Cable calls (LF): {n_lf_call_ids}")
print(f"South Cable calls (HF): {s_hf_call_ids}")
print(f"South Cable calls (LF): {s_lf_call_ids}")

# +

fig = dw.assoc.plot_associated_bicable_paper(peaks, n_longi_offset, pair_assoc, pair_loc, associations, localizations, n_cable_pos, s_cable_pos, n_dist, s_dist, selected_channels_m, dx, c0, fs, height_ratio, title='baseline (60 it.)')
fig.savefig('../figs/Figure6bis_b.pdf', bbox_inches=None, transparent=True)
plt.show()
# -

# # Far window associations

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

fig = dw.assoc.plot_associated_bicable_paper(peaks, n_longi_offset, pair_assoc, pair_loc, associations, localizations, n_cable_pos, s_cable_pos, n_dist, s_dist, selected_channels_m, dx, c0, fs, height_ratio, title='FW (60 it.)')
fig.savefig('../figs/Figure6bis_c.pdf', bbox_inches=None, transparent=True)
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
# -

peaks = (npeakshf, npeakslf, speakshf, speakslf)
fig = dw.assoc.plot_associated_bicable_paper(peaks, n_longi_offset, pair_assoc, pair_loc, associations, localizations, n_cable_pos, s_cable_pos, n_dist, s_dist, selected_channels_m, dx, c0, fs, height_ratio, title='Gabor + FW (60 it.)')
fig.savefig('../figs/Figure6bis_d.pdf', bbox_inches=None, transparent=True)
plt.show()

# +
# # Check annotation data structure
# print("North annotations:")
# print(n_annot.head())
# print("\nColumns:", n_annot.columns.tolist())
# print("\nSouth annotations:")
# print(s_annot.head())
# print("\nColumns:", s_annot.columns.tolist())
