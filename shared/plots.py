import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


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
