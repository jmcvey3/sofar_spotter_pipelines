import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from mhkit.tidal import graphics


def wave_spectra(dataset):
    # Wave spectra figure
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.subplots_adjust(left=0.14, right=0.95, top=0.95, bottom=0.1)

    # Organize by wave height
    hs_max = dataset["wave_hs"].max()
    step = hs_max / 10
    bins = np.arange(0, hs_max + step, step)
    spec_U = (
        dataset["wave_energy_density"]
        .assign_coords({"time": dataset["wave_hs"].values})
        .rename({"time": "height"})
    )
    grouped_spec = spec_U.groupby_bins("height", bins).mean()
    # Create colormap
    norm = plt.Normalize()
    colors = plt.cm.turbo(norm(bins))
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)

    for i in range(len(bins) - 1):
        ax.loglog(dataset["frequency"], grouped_spec[i], c=colors[i])
    fig.colorbar(sm, ax=ax, label="Sig Wave Height [m]")
    plt.grid()
    m = -4
    x = np.logspace(-1, 0)
    y = 10 ** (-4) * x**m
    ax.loglog(x, y, "--", c="black", label="f^-4")
    ax.set(
        ylim=(0.0005, 1),
        xlabel="Frequency [Hz]",
        ylabel="Energy Density [m^2/Hz]",
    )

    return fig, ax


def directional_spectra(spectrum: xr.DataArray):
    # Plot Fourier directional spectra
    fig, ax = plt.subplots(
        figsize=(8, 6), subplot_kw=dict(projection="polar"), constrained_layout=True
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # Use frequencies up to 0.5 Hz
    # spectrum = dataset["wave_dir_energy_density"].mean("time")
    # Create grid and plot
    a, f = np.meshgrid(np.deg2rad(spectrum["direction"]), 1 / spectrum["frequency"])
    color_level_max = np.nanmax(spectrum.values)
    levels = np.linspace(0, color_level_max, 11)
    c = ax.contourf(a, f, spectrum, levels=levels, cmap="Blues")
    cbar = plt.colorbar(c)
    cbar.set_label(r"ESD [m$^2$s/deg]", labelpad=20)
    ax.set_ylim(2, 12)
    ylabels = ax.get_yticklabels()
    ylabels = [ilabel.get_text() for ilabel in ax.get_yticklabels()]
    ylabels = [ilabel + " s" for ilabel in ylabels]
    ticks_loc = ax.get_yticks()
    ax.set_yticks(ticks_loc)
    ax.set_yticklabels(ylabels)

    return fig, ax


def wavelets(dataset):
    ## Wavelet figure
    fig, ax = plt.subplots(
        figsize=(10, 5), subplot_kw={"yscale": "log"}, constrained_layout=True
    )
    vmax = 0.35
    pcm = ax.pcolormesh(
        dataset["time"].values,
        dataset["frequency"].values,
        dataset["wavelet_energy_density"].T,
        cmap="Blues",
        shading="nearest",
        vmin=0,
        vmax=vmax,
    )
    ax.set(ylim=(0.03, 1), ylabel="Frequency [Hz]", xlabel="Time")
    fig.colorbar(pcm, ax=ax, label=r"Wavelet Energy Density [m$^2$]")
    # Quiver arrows show propagation direction (wave_direction is "from" convention, so flip 180)
    theta = np.deg2rad((dataset["wave_direction"] + 180) % 360)
    qu = np.sin(theta).T
    qv = np.cos(theta).T
    time_grid, freq_grid = np.meshgrid(
        dataset["time"].values, dataset["frequency"].values
    )
    step_t, step_f = 10, 5  # subsample to avoid a cluttered quiver
    energy = dataset["wavelet_energy_density"].T.values[::step_f, ::step_t]
    energy_thresh = 0.1 * vmax  # only show arrows above 10% of max energy shown
    qu_masked = np.where(energy >= energy_thresh, qu.values[::step_f, ::step_t], np.nan)
    qv_masked = np.where(energy >= energy_thresh, qv.values[::step_f, ::step_t], np.nan)
    ax.quiver(
        time_grid[::step_f, ::step_t],
        freq_grid[::step_f, ::step_t],
        qu_masked,
        qv_masked,
        color="black",
        scale=60,
        width=0.002,
    )

    return fig, ax


def wave_rose(dataset):
    # Plot wave roses
    fig, ax = plt.subplots(
        figsize=(8, 6), subplot_kw={"projection": "polar"}, constrained_layout=True
    )
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    # Use 360 degrees
    dp = dataset["wave_dp"].copy(deep=True).values
    dp = dp % 360
    # Calculate the 2D histogram
    H, dir_edges, vel_edges = graphics._histogram(dp, dataset["wave_hs"], 10, 0.5)
    # Determine number of bins
    dir_bins = H.shape[0]
    h_bins = H.shape[1]
    # Create the angles
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / dir_bins)
    # Set bar color based on wind speed
    colors = plt.cm.Wistia(np.linspace(0, 1.0, h_bins))
    # Set the current speed bin label names
    # Calculate the 2D histogram
    labels = [f"{i:.1f}-{j:.1f}" for i, j in zip(vel_edges[:-1], vel_edges[1:])]
    # Initialize the vertical-offset (polar radius) for the stacked bar chart.
    r_offset = np.zeros(dir_bins)
    for h_bin in range(h_bins):
        # Plot fist set of bars in all directions
        ax.bar(
            thetas,
            H[:, h_bin],
            width=(2 * np.pi / dir_bins),
            bottom=r_offset,
            color=colors[h_bin],
            label=labels[h_bin],
        )
        # Increase the radius offset in all directions
        r_offset = r_offset + H[:, h_bin]
    # Add the a legend for current speed bins
    plt.legend(loc="best", title="Hs [m]", bbox_to_anchor=(1.29, 1.00), ncol=1)
    # Get the r-ticks (polar y-ticks)
    yticks = plt.yticks()
    # Format y-ticks with  units for clarity
    rticks = [f"{y:.1f}%" for y in yticks[0]]
    # Set the y-ticks
    ax.set_yticks(yticks[0], rticks)

    return fig, ax


def plot_gps(dataset):
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.subplots_adjust(left=0.16, right=0.95, top=0.95, bottom=0.17)
    ax.scatter(dataset["longitude"], dataset["latitude"])
    ax.set(ylabel="Latitude [deg N]", xlabel="Longitude [deg E]")
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    # ax.set(
    #     xlim=(dataset.geospatial_lon_min, dataset.geospatial_lon_max),
    #     ylim=(dataset.geospatial_lat_min, dataset.geospatial_lat_max),
    # )
    # Set grid below
    ax.set_axisbelow(True)
    ax.grid()
    ax.tick_params(labelrotation=45)
    return fig, ax
