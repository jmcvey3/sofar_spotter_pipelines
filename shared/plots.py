import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


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
