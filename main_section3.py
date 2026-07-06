## Section III.A-C code
# ## This code doesn't plot the last figures of Section III.C for RAM management and computationnal efficiency

# Imports
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import das4whales as dw
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import RegularGridInterpolator

plt.rcParams["font.size"] = 24
plt.rcParams["lines.linewidth"] = 3


def main():
    os.makedirs("figs", exist_ok=True)

    # Load the peak indexes and the metadata
    # data directory (relative to repo root)
    directory = "data/detections/"
    # For Gabor filtered detections:
    # directory = 'data/detections_Gabor/'

    n_ds = xr.load_dataset(
        os.path.join(
            directory, "peaks_indexes_tp_North_2021-11-04_02:00:02_ipi3_th_4.nc"
        )
    )
    s_ds = xr.load_dataset(
        os.path.join(
            directory, "peaks_indexes_tp_South_2021-11-04_02:00:02_ipi3_th_5.nc"
        )
    )

    # Constants from the metadata
    fs = n_ds.attrs["fs"]
    dx = n_ds.attrs["dx"]
    nnx = n_ds.attrs["data_shape"][0]
    snx = s_ds.attrs["data_shape"][0]
    n_selected_channels_m = n_ds.attrs["selected_channels_m"]
    s_selected_channels_m = s_ds.attrs["selected_channels_m"]

    # Constants management
    c0 = 1480
    n_selected_channels = dw.data_handle.get_selected_channels(
        n_selected_channels_m, dx
    )
    s_selected_channels = dw.data_handle.get_selected_channels(
        s_selected_channels_m, dx
    )
    n_begin_chan = n_selected_channels[0]
    n_end_chan = n_selected_channels[1]
    n_longi_offset = n_selected_channels[0] // n_selected_channels[2]
    s_begin_chan = s_selected_channels[0]
    s_end_chan = s_selected_channels[1]
    s_longi_offset = s_selected_channels[0] // s_selected_channels[2]
    n_dist = (np.arange(nnx) * n_selected_channels[2] + n_selected_channels[0]) * dx
    s_dist = (np.arange(snx) * s_selected_channels[2] + s_selected_channels[0]) * dx
    dx = dx * n_selected_channels[2]

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

    # Sort the peaks based on SNR difference
    npeakshf, nSNRhf, npeakslf, nSNRlf = dw.detect.resolve_hf_lf_crosstalk(
        npeakshf, npeakslf, nSNRhf, nSNRlf, dt_tol=100, dx_tol=30
    )

    speakshf, sSNRhf, speakslf, sSNRlf = dw.detect.resolve_hf_lf_crosstalk(
        speakshf, speakslf, sSNRhf, sSNRlf, dt_tol=100, dx_tol=30
    )

    # ## Plot the map
    # Import the cable location
    df_north = pd.read_csv("data/north_DAS_multicoord.csv")
    df_south = pd.read_csv("data/south_DAS_multicoord.csv")

    # Extract the part of the dataframe used for the time picking process
    idx_shift0 = int(
        n_begin_chan - df_north["chan_idx"].iloc[0]
    )  # Shift between the cable locations (starting at the beach) and the channel locations
    idx_shiftn = int(n_end_chan - df_north["chan_idx"].iloc[-1])

    df_north_used = df_north.iloc[idx_shift0 : idx_shiftn : n_selected_channels[2]][
        :nnx
    ]

    idx_shift0 = int(
        s_begin_chan - df_south["chan_idx"].iloc[0]
    )  # Shift between the cable locations (starting at the beach) and the channel locations
    idx_shiftn = int(s_end_chan - df_south["chan_idx"].iloc[-1])

    df_south_used = df_south.iloc[idx_shift0 : idx_shiftn : s_selected_channels[2]][
        :snx
    ]

    # Import the bathymetry data
    bathy, xlon, ylat = dw.map.load_bathymetry("data/GMRT_OOI_RCA_Cables.grd")
    print(f"Origin of the corrdinates. Latitude = {ylat[0]}, Longitude = {xlon[-1]}")

    utm_x0, utm_y0 = dw.map.latlon_to_utm(xlon[0], ylat[0])
    utm_xf, utm_yf = dw.map.latlon_to_utm(xlon[-1], ylat[-1])

    # Change the reference point to the last point
    x0, y0 = utm_xf - utm_x0, utm_y0 - utm_y0
    xf, yf = utm_xf - utm_xf, utm_yf - utm_y0
    print(xf, yf)

    # # Create vectors of coordinates
    x = np.linspace(x0, xf, len(xlon))
    y = np.linspace(y0, yf, len(ylat))

    # Cable geometry (make it correspond to x,y,z = cable_pos[:, 0], cable_pos[:, 1], cable_pos[:, 2])
    n_cable_pos = np.zeros((len(df_north_used), 3))
    s_cable_pos = np.zeros((len(df_south_used), 3))

    n_cable_pos[:, 0] = df_north_used["x"]
    n_cable_pos[:, 1] = df_north_used["y"]
    n_cable_pos[:, 2] = df_north_used["depth"]

    s_cable_pos[:, 0] = df_south_used["x"]
    s_cable_pos[:, 1] = df_south_used["y"]
    s_cable_pos[:, 2] = df_south_used["depth"]

    # Create a grid of coordinates, choosing the spacing of the grid
    dx_grid = 2000  # [m]
    dy_grid = 2000  # [m]
    xg, yg = np.meshgrid(np.arange(xf, x0, dx_grid), np.arange(y0, yf, dy_grid))

    ti = 0
    zg = -30

    interpolator = RegularGridInterpolator((x, y), bathy.T)
    bathy_interp = interpolator((xg, yg))

    # Remove points if the ocean depth is too shallow (i.e., less than -25 m)
    mask = bathy_interp < -25
    # Compute arrival times only for valid grid points
    # Flatten the grid points
    xg, yg = xg[mask], yg[mask]

    # In case of a meshgrid object (non flattened), use the following code:
    # xg[~mask] = np.nan
    # yg[~mask] = np.nan

    # Compute KDEs for all delayed picks
    # TODO: KDE number of points proportional to the number of picks (y-axis)?
    dt_kde = 0.5  # [s] Time resolution of the KDE (overlap)
    bin_width = 1
    n_shape_x = xg.shape[0]
    s_shape_x = xg.shape[0]
    dt_sel = 1.4  # [s] Selected time "distance" from the theoretical arrival time

    n_up_peaks_hf = np.copy(npeakshf)
    s_up_peaks_hf = np.copy(speakshf)
    n_up_peaks_lf = np.copy(npeakslf)
    s_up_peaks_lf = np.copy(speakslf)
    n_arr_tg = dw.loc.calc_arrival_times(ti, n_cable_pos, (xg, yg, zg), c0)
    s_arr_tg = dw.loc.calc_arrival_times(ti, s_cable_pos, (xg, yg, zg), c0)

    print(n_arr_tg.shape, nnx, n_cable_pos.shape)
    print(s_arr_tg.shape, snx, s_cable_pos.shape)

    n_idx_times_hf = np.array(n_up_peaks_hf[1]) / fs  # Update with the remaining peaks
    n_idx_times_lf = np.array(n_up_peaks_lf[1]) / fs  # Update with the remaining peaks
    s_idx_times_hf = np.array(s_up_peaks_hf[1]) / fs  # Update with the remaining peaks
    s_idx_times_lf = np.array(s_up_peaks_lf[1]) / fs  # Update with the remaining peaks

    # Make a delayed picks array for all the grid points
    # Broadcast the time indices delayed by the theoretical arrival times for the grid points

    n_delayed_picks_hf = n_idx_times_hf[None, :] - n_arr_tg[:, n_up_peaks_hf[0]]
    n_delayed_picks_lf = n_idx_times_lf[None, :] - n_arr_tg[:, n_up_peaks_lf[0]]
    s_delayed_picks_hf = s_idx_times_hf[None, :] - s_arr_tg[:, s_up_peaks_hf[0]]
    s_delayed_picks_lf = s_idx_times_lf[None, :] - s_arr_tg[:, s_up_peaks_lf[0]]

    global_min = min(
        np.min(n_delayed_picks_hf),
        np.min(n_delayed_picks_lf),
        np.min(s_delayed_picks_hf),
        np.min(s_delayed_picks_lf),
    )
    global_max = max(
        np.max(n_delayed_picks_hf),
        np.max(n_delayed_picks_lf),
        np.max(s_delayed_picks_hf),
        np.max(s_delayed_picks_lf),
    )
    Nkde = np.ceil((global_max - global_min) / dt_kde).astype(int) + 1
    t_kde = np.linspace(global_min, global_max, Nkde)

    n_kde_hf = np.array(
        Parallel(n_jobs=-1)(
            delayed(dw.assoc.fast_kde_rect)(
                n_delayed_picks_hf[i, :],
                t_kde,
                overlap=dt_kde,
                bin_width=bin_width,
                weights=nSNRhf,
            )
            for i in range(n_shape_x)
        )
    )
    n_kde_lf = np.array(
        Parallel(n_jobs=-1)(
            delayed(dw.assoc.fast_kde_rect)(
                n_delayed_picks_lf[i, :],
                t_kde,
                overlap=dt_kde,
                bin_width=bin_width,
                weights=nSNRlf,
            )
            for i in range(n_shape_x)
        )
    )
    s_kde_hf = np.array(
        Parallel(n_jobs=-1)(
            delayed(dw.assoc.fast_kde_rect)(
                s_delayed_picks_hf[i, :],
                t_kde,
                overlap=dt_kde,
                bin_width=bin_width,
                weights=sSNRhf,
            )
            for i in range(s_shape_x)
        )
    )
    s_kde_lf = np.array(
        Parallel(n_jobs=-1)(
            delayed(dw.assoc.fast_kde_rect)(
                s_delayed_picks_lf[i, :],
                t_kde,
                overlap=dt_kde,
                bin_width=bin_width,
                weights=sSNRlf,
            )
            for i in range(s_shape_x)
        )
    )

    print(n_kde_hf.shape, n_kde_lf.shape)
    print(n_delayed_picks_hf.shape, n_delayed_picks_lf.shape)
    print(s_kde_hf.shape, s_kde_lf.shape)
    print(s_delayed_picks_hf.shape, s_delayed_picks_lf.shape)

    hf_kde = n_kde_hf + s_kde_hf
    lf_kde = n_kde_lf + s_kde_lf

    # Find the maximum for the 4 kde sets

    n_max_kde_hf = np.argmax(n_kde_hf)
    nhf_imax, nhf_tmax = np.unravel_index(n_max_kde_hf, n_kde_hf.shape)

    n_max_kde_lf = np.argmax(n_kde_lf)
    nlf_imax, nlf_tmax = np.unravel_index(n_max_kde_lf, n_kde_lf.shape)

    s_max_kde_hf = np.argmax(s_kde_hf)
    shf_imax, shf_tmax = np.unravel_index(s_max_kde_hf, s_kde_hf.shape)

    s_max_kde_lf = np.argmax(s_kde_lf)
    slf_imax, slf_tmax = np.unravel_index(s_max_kde_lf, s_kde_lf.shape)

    print(
        f"North HF max kde: {n_max_kde_hf}, max index: {nhf_imax}, max time: {nhf_tmax}"
    )
    print(
        f"North LF max kde: {n_max_kde_lf}, max index: {nlf_imax}, max time: {nlf_tmax}"
    )
    print(
        f"South HF max kde: {s_max_kde_hf}, max index: {shf_imax}, max time: {shf_tmax}"
    )
    print(
        f"South LF max kde: {s_max_kde_lf}, max index: {slf_imax}, max time: {slf_tmax}"
    )

    # Find the maximum for the 2 combined kde sets
    hf_max_kde = np.argmax(hf_kde)
    hf_imax, hf_tmax = np.unravel_index(hf_max_kde, hf_kde.shape)

    lf_max_kde = np.argmax(lf_kde)
    lf_imax, lf_tmax = np.unravel_index(lf_max_kde, lf_kde.shape)

    print(
        f"Combined HF max kde: {hf_max_kde}, max index: {hf_imax}, max time: {hf_tmax}"
    )
    print(
        f"Combined LF max kde: {lf_max_kde}, max index: {lf_imax}, max time: {lf_tmax}"
    )

    # Print the delayed time for the maximum KDE on north and south cables, lf
    # Calculate height ratios based on y-range
    y_range_north = n_selected_channels_m[1] - n_selected_channels_m[0]  # meters
    y_range_south = s_selected_channels_m[1] - s_selected_channels_m[0]  # meters
    height_ratio = y_range_south / y_range_north
    print(height_ratio)

    # Calculate the common x-range for all subplots
    x_min = min(
        min(s_delayed_picks_lf[lf_imax, :]), min(n_delayed_picks_lf[lf_imax, :])
    )
    x_max = max(
        max(s_delayed_picks_lf[lf_imax, :]), max(n_delayed_picks_lf[lf_imax, :])
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(10, 16),
        constrained_layout=True,
        sharex=True,
        sharey=False,
        gridspec_kw={"height_ratios": [1, 0.3, height_ratio, 0.3]},
    )
    axes[0].set_title("North Cable")
    axes[0].scatter(
        n_delayed_picks_hf[hf_imax, :],
        (n_longi_offset + npeakshf[0][:]) * dx * 1e-3,
        label="HF",
        c=nSNRhf,
        s=nSNRhf * 0.8,
        cmap="plasma",
        rasterized=True,
    )
    axes[0].scatter(
        n_delayed_picks_lf[lf_imax, :],
        (n_longi_offset + npeakslf[0][:]) * dx * 1e-3,
        label="LF",
        c=nSNRlf,
        s=nSNRlf * 0.8,
        cmap="viridis",
        rasterized=True,
    )
    axes[0].set_xlim(x_min, x_max)
    axes[0].grid(linestyle="--", alpha=0.5)
    axes[0].set_ylabel("Distance [km]")
    axes[0].set_aspect("equal", adjustable="datalim")

    axes[1].plot(t_kde, n_kde_hf[nhf_imax, :], color="tab:orange", lw=3, label="HF")
    axes[1].plot(t_kde, n_kde_lf[nlf_imax, :], color="tab:green", lw=3, label="LF")
    axes[1].set_ylim(0, max(np.max(n_kde_hf), np.max(n_kde_lf)) * 1.1)
    axes[1].grid(linestyle="--", alpha=0.5)
    axes[1].legend()
    axes[1].ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    axes[1].set_ylabel("KDE [-]")
    axes[1].legend(loc="lower left")

    axes[2].set_title("South Cable")
    axes[2].scatter(
        s_delayed_picks_hf[hf_imax, :],
        (s_longi_offset + speakshf[0][:]) * dx * 1e-3,
        label="HF",
        c=sSNRhf,
        s=sSNRhf * 0.8,
        cmap="plasma",
        rasterized=True,
    )
    axes[2].scatter(
        s_delayed_picks_lf[lf_imax, :],
        (s_longi_offset + speakslf[0][:]) * dx * 1e-3,
        label="LF",
        c=sSNRlf,
        s=sSNRlf * 0.8,
        cmap="viridis",
        rasterized=True,
    )
    axes[2].grid(linestyle="--", alpha=0.5)
    axes[2].set_ylabel("Distance [km]")
    axes[2].set_aspect("equal", adjustable="datalim")

    axes[3].plot(t_kde, s_kde_hf[hf_imax, :], color="tab:orange", lw=3, label="HF")
    axes[3].plot(t_kde, s_kde_lf[lf_imax, :], color="tab:green", lw=3, label="LF")
    axes[3].set_ylim(0, max(np.max(s_kde_hf), np.max(s_kde_lf)) * 1.1)
    axes[3].ticklabel_format(style="scientific", axis="y", scilimits=(0, 0))
    axes[3].set_ylabel("KDE [-]")
    axes[3].set_xlabel("Delayed time [s]")
    axes[3].legend(loc="lower left")

    plt.grid(linestyle="--", alpha=0.5)
    fig.savefig("figs/Figure3.pdf", bbox_inches="tight", transparent=True, format="pdf")
    # plt.show()()

    max_time_hf = t_kde[hf_tmax]
    max_time_lf = t_kde[lf_tmax]

    # Plot the hyberbola on top of the picks
    # Create figure
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 16),
        sharex=True,
        sharey=False,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1, height_ratio]},
    )

    # First subplot
    axes[0].scatter(
        npeakshf[1][:] / fs,
        (n_selected_channels_m[0] + npeakshf[0][:] * dx) * 1e-3,
        c="grey",
        s=nSNRhf,
        rasterized=True,
        alpha=0.7,
    )
    axes[0].scatter(
        npeakslf[1][:] / fs,
        (n_selected_channels_m[0] + npeakslf[0][:] * dx) * 1e-3,
        c="grey",
        s=nSNRlf,
        rasterized=True,
        alpha=0.7,
    )

    axes[0].plot(
        max_time_hf + n_arr_tg[hf_imax, :],
        n_dist / 1e3,
        ls="-",
        lw=3,
        color="tab:orange",
        label="HF",
    )
    axes[0].fill_betweenx(
        n_dist / 1e3,
        max_time_hf + n_arr_tg[hf_imax, :] - dt_sel,
        max_time_hf + n_arr_tg[hf_imax, :] + dt_sel,
        color="tab:orange",
        alpha=0.3,
        edgecolor="tab:orange",
        linewidth=1,
    )

    axes[0].plot(
        max_time_lf + n_arr_tg[lf_imax, :],
        n_dist / 1e3,
        ls="-",
        lw=3,
        color="tab:green",
        label="LF",
    )
    axes[0].fill_betweenx(
        n_dist / 1e3,
        max_time_lf + n_arr_tg[lf_imax, :] - dt_sel,
        max_time_lf + n_arr_tg[lf_imax, :] + dt_sel,
        color="tab:green",
        alpha=0.3,
        edgecolor="tab:green",
        linewidth=1,
    )

    axes[0].set_title("North Cable")
    axes[0].set_ylabel("Distance [km]")
    axes[0].grid(linestyle="--", alpha=0.5)
    axes[0].set_ylim(min(n_dist / 1e3), max(n_dist / 1e3))
    axes[0].set_aspect("equal", adjustable="box")

    # Second subplot
    axes[1].scatter(
        speakshf[1][:] / fs,
        (s_selected_channels_m[0] + speakshf[0][:] * dx) * 1e-3,
        c="grey",
        s=sSNRhf,
        rasterized=True,
        alpha=0.7,
    )
    axes[1].scatter(
        speakslf[1][:] / fs,
        (s_selected_channels_m[0] + speakslf[0][:] * dx) * 1e-3,
        c="grey",
        s=sSNRlf,
        rasterized=True,
        alpha=0.7,
    )

    axes[1].plot(
        max_time_hf + s_arr_tg[hf_imax, :],
        s_dist / 1e3,
        ls="-",
        lw=3,
        color="tab:orange",
        label="HF",
    )
    axes[1].fill_betweenx(
        s_dist / 1e3,
        max_time_hf + s_arr_tg[hf_imax, :] - dt_sel,
        max_time_hf + s_arr_tg[hf_imax, :] + dt_sel,
        color="tab:orange",
        alpha=0.5,
        edgecolor="tab:orange",
        linewidth=1,
    )

    axes[1].plot(
        max_time_lf + s_arr_tg[lf_imax, :],
        s_dist / 1e3,
        ls="-",
        lw=3,
        color="tab:green",
        label="LF",
    )
    axes[1].fill_betweenx(
        s_dist / 1e3,
        max_time_lf + s_arr_tg[lf_imax, :] - dt_sel,
        max_time_lf + s_arr_tg[lf_imax, :] + dt_sel,
        color="tab:green",
        alpha=0.5,
        edgecolor="tab:green",
        linewidth=1,
    )

    axes[1].set_title("South Cable")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Distance [km]")
    axes[1].grid(linestyle="--", alpha=0.5)
    # set xlim to the same as the first subplot
    axes[1].set_xlim(min(npeakshf[1][:] / fs), max(npeakshf[1][:] / fs))
    axes[1].set_ylim(min(s_dist / 1e3), max(s_dist / 1e3))
    axes[1].set_xticks(np.arange(0, max(speakshf[1][:] / fs) + 10, 10))
    axes[1].set_aspect("equal", adjustable="box")

    for ax in axes:
        ax.legend(loc="best", frameon=True, fancybox=True, shadow=True)

    plt.savefig("figs/Figure5.pdf", bbox_inches="tight", transparent=True, format="pdf")
    # plt.show()()


if __name__ == "__main__":
    main()
