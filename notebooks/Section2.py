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
import matplotlib.pyplot as plt
import das4whales as dw
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
from tqdm import tqdm
import dask.array as da
import xarray as xr
# from dask import delayed
from joblib import Parallel, delayed
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
plt.rcParams['font.size'] = 24

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
# Create two list of coordinates, for ponts every 10 km along the cables, the spatial resolution is 2m 
opticald_n = []
opticald_s = []
disp_step = 10000 # [m]
dx_ch = 2.0419 # [m]
idx_step = int(disp_step / dx_ch)

for i in range(int(idx_step-df_north["chan_idx"].iloc[0]), len(df_north), int(10000/2)):
    opticald_n.append((df_north['x'][i], df_north['y'][i]))

for i in range(int(idx_step-df_north["chan_idx"].iloc[0]), len(df_south), int(10000/2)):
    opticald_s.append((df_south['x'][i], df_south['y'][i]))

# +
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

# Compute arrival times for the grid
n_arr_tg = dw.loc.calc_arrival_times(ti, n_cable_pos, (xg, yg, zg), c0)
s_arr_tg = dw.loc.calc_arrival_times(ti, s_cable_pos, (xg, yg, zg), c0)

# +
# Convert Pacific City latitude and longitude to UTM coordinates 
lat_pc, lon_pc = 45.201801, -123.960861 # Pacific City lat/lon
utm_x_pc, utm_y_pc = dw.map.latlon_to_utm(lon_pc, lat_pc)

# Adjust Pacific City's UTM coordinates to the shifted coordinate system (relative to utm_xf, utm_yf)
utm_x_pc, utm_y_pc = utm_xf - utm_x_pc, utm_y_pc - utm_y0

# +
# Pick grid location to plot as examples
examples = [421, 510, 800]  # Example indices to plot
colors = ['tab:pink', 'tab:purple', 'tab:green'] # cm.plasma(np.linspace(0, 0.75, len(examples))) # Color map for examples

print(f"Selected examples to plot on grid and delays: {examples}")

# +
# Plot the grid points on the map
import cmocean.cm as cmo
from matplotlib.ticker import FuncFormatter
colors_undersea = cmo.deep_r(np.linspace(0, 1, 256)) # blue colors for under the sea
colors_land = np.array([[0.5, 0.5, 0.5, 1]])  # Solid gray for above sea level

# Combine the color maps
all_colors = np.vstack((colors_undersea, colors_land))
custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', all_colors)

extent = [x[0], x[-1], y[0], y[-1]]

print(f'Extent of the map: {extent} km')    

# Set the light source
ls = LightSource(azdeg=350, altdeg=45)

plt.figure(figsize=(14, 7))
ax = plt.gca()

# Plot the bathymetry relief in background
rgb = ls.shade(bathy, cmap=custom_cmap, vert_exag=0.1, blend_mode='overlay', vmin=np.min(bathy), vmax=0)
plot = ax.imshow(rgb, extent=extent, aspect='equal', origin='lower' , vmin=np.min(bathy), vmax=0)

# Plot the cable location in 2D
ax.plot(df_north['x'], df_north['y'], 'tab:red', label='North cable', lw=2.5)
ax.plot(df_south['x'], df_south['y'], 'tab:orange', label='South cable', lw=2.5)

# Plot the used cable locations
# ax.plot(df_north_used['x'], df_north_used['y'], 'tab:green', label='Used cable locations')

# Plot the grid points
ax.scatter(xg, yg, c='grey', s=10, label='Grid points')
# Plot the examples grid points
for i, example in enumerate(examples):
    if i == 0:
        ax.scatter(xg[example], yg[example], color='white', edgecolors='k', s=200, label=f'Example sources', marker='*')
        ax.scatter(xg[example], yg[example], color=colors[i], s=300, marker='*', edgecolors='k', zorder=5)
    else:
        ax.scatter(xg[example], yg[example], color=colors[i], s=300, marker='*', edgecolors='k', zorder=5)

# Plot points along the cable every 10 km in terms of optical distance
for i, point in enumerate(opticald_n, start=1):
    if i == 1:
        ax.plot(point[0], point[1], '.', color='k', markersize=10, label='Cable Length [km]')
        # Annotate the points with the distance
        ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, 10), ha='center')
    else:
        ax.plot(point[0], point[1], '.', color='k', markersize=10)
        # Annotate the points with the distance
        ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, 10), ha='center')

for i, point in enumerate(opticald_s, start=1):
    ax.plot(point[0], point[1], '.', color='k', markersize=10)
    ax.annotate(f'{i*10}', (point[0], point[1]), textcoords='offset points', xytext=(5, -30), ha='center')

# Plot the Pacific City location
ax.scatter(utm_x_pc, utm_y_pc, marker='s', color='tab:red', s=100, zorder=5, edgecolor='k')
ax.annotate('Pacific City', (utm_x_pc, utm_y_pc), 
            textcoords='offset points', xytext=(15, 20), 
            ha='center', color='k',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.8))

# Plot the repeater locations
ax.scatter(df_north['x'].iloc[-1], df_north['y'].iloc[-1], marker='o', color='white', s=100, zorder=5, edgecolor='k', label='Repeaters')
ax.scatter(df_north['x'].iloc[-1], df_north['y'].iloc[-1], marker='o', color='tab:red', s=100, zorder=5, edgecolor='k')
ax.scatter(df_south['x'].iloc[-1], df_south['y'].iloc[-1], marker='o', color='tab:orange', s=100, zorder=5, edgecolor='k')

# Add dashed contours at selected depths with annotations
# depth_levels = [-20]

# contour_dashed = ax.contour(bathy, levels=depth_levels, colors='k', linestyles='--', extent=extent, alpha=0.6)
# ax.clabel(contour_dashed, fmt='%d m', inline=True, fontsize=9)

# Use a proxy artist for the color bar
im = ax.imshow(bathy, cmap=custom_cmap, extent=extent, aspect='equal', origin='lower', vmin=np.min(bathy), vmax=0)
im_ratio = bathy.shape[1] / bathy.shape[0]
plt.colorbar(im, ax=ax, label='Depth [m]', pad=0.02, orientation='vertical', aspect=25, fraction=0.0195)
im.remove()
# Set the labels

# Convert axis labels to kilometers using custom formatter
def m_to_km_formatter(x, pos):
    """Convert meters to kilometers for axis labels"""
    return f'{x/1000:.0f}'

# Apply the formatter to both axes
ax.xaxis.set_major_formatter(FuncFormatter(m_to_km_formatter))
ax.yaxis.set_major_formatter(FuncFormatter(m_to_km_formatter))

plt.xlabel('x [km]')
plt.ylabel('y [km]')
plt.legend(loc='upper left', labelspacing=0.2, ncol=2, columnspacing=0.6)
plt.tight_layout()
plt.savefig('../figs/Figure1a.pdf', bbox_inches='tight', transparent=True)
plt.show()

# +
# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3, 7), sharex=True, constrained_layout=True)

# North
ax1.set_title('North Cable')
for i in range(xg.shape[0]):
    ax1.plot(n_arr_tg[i, :], n_dist/1e3, lw=0.5, color='tab:blue', alpha=0.1)
for j, i in enumerate(examples):
    ax1.plot(n_arr_tg[i, :], n_dist/1e3, lw=2, color=colors[j])
ax1.set_ylabel('Distance [km]')
ax1.spines[['top', 'right']].set_visible(False)
ax1.grid(ls='--', alpha=0.5)
ax1.set_aspect('equal')

# South
ax2.set_title('South Cable')
for i in range(xg.shape[0]):
    ax2.plot(s_arr_tg[i, :], s_dist/1e3, lw=0.5, color='tab:blue', alpha=0.1)
for j, i in enumerate(examples):
    ax2.plot(s_arr_tg[i, :], s_dist/1e3, lw=2, color=colors[j])
ax2.set_ylabel('Distance [km]')
ax2.set_xlabel('Time [s]')
ax2.spines[['top', 'right']].set_visible(False)
ax2.grid(ls='--', alpha=0.5)
ax2.set_aspect('equal')

# plt.tight_layout()
plt.savefig('../figs/Figure1b.pdf', bbox_inches='tight', transparent=True)
plt.show()
