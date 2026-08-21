import numpy as np
import xarray as xr
from typing import Dict
from tsdat import TransformationPipeline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cmocean.cm import amp_r, dense, haline

from shared.wave_analysis import constants, wave_analysis


class VapWaves(TransformationPipeline):
    """---------------------------------------------------------------------------------
    VAP pipeline for calculating wave statistics from a Sofar Spotter wave buoy data.
    ---------------------------------------------------------------------------------"""

    def hook_customize_input_datasets(self, input_datasets) -> Dict[str, xr.Dataset]:
        # Code hook to customize any input datasets prior to datastreams being combined
        # and data converters being run.

        # Need to write in frequency coordinate that will be used later
        for key in input_datasets:
            if "pos" in key:
                # Create FFT frequency vector
                nfft = constants["fs"] * constants["wat"] // constants["fft_decimation"]
                f = np.fft.fftfreq(int(nfft), 1 / constants["fs"])
                # Use only positive frequencies
                freq = np.abs(f[1 : int(nfft / 2.0 + 1)])
                # Trim frequency vector to > 0.0455 Hz (wave periods between 1 and 22 s)
                freq = freq[
                    np.where(
                        (freq > constants["freq_slc"][0])
                        & (freq <= constants["freq_slc"][1])
                    )
                ]
                input_datasets[key] = input_datasets[key].assign_coords(
                    {"frequency": freq, "direction": constants["dir_centers"]}
                )

                return input_datasets

    def hook_customize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset before qc is applied

        ds = wave_analysis(dataset)

        return ds.drop_vars(("x", "y", "z"))

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset after qc is applied
        # but before it gets saved to the storage area
        return dataset

    def hook_plot_dataset(self, dataset: xr.Dataset):
        # (Optional, recommended) Create plots.
        plt.style.use("default")  # clear any styles that were set before

        # Wave spectra figure
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(left=0.14, right=0.95, top=0.95, bottom=0.1)
        for timestamp in dataset["time"]:
            ax.loglog(
                dataset["frequency"],
                dataset["wave_energy_density"].sel(time=timestamp),
                label="vertical",
            )
        m = -4
        x = np.logspace(-1, 0)
        y = 10 ** (-4) * x**m
        ax.loglog(x, y, "--", c="black", label="f^-4")
        ax.set(
            ylim=(0.0001, 10),
            xlabel="Frequency [Hz]",
            ylabel="Energy Density [m^2/Hz]",
        )
        plot_file = self.get_ancillary_filepath(title="elevation_spectrum")
        fig.savefig(plot_file)

        # Wave time-series figure
        fig, ax = plt.subplots(3, 1, figsize=(10, 7), constrained_layout=True)
        ax[0].plot(
            dataset["time"],
            dataset["wave_hs"],
            ".-",
            label="Significant Wave Height",
            color=amp_r(0.10),
        )
        ax[0].set(ylabel="Height [m]")

        ax[1].plot(
            dataset["time"],
            dataset["wave_ta"],
            ".-",
            label="Mean Period",
            color=dense(0.15),
        )
        ax[1].plot(
            dataset["time"],
            dataset["wave_tp"],
            ".-",
            label="Peak Period",
            color=dense(0.35),
        )
        ax[1].plot(
            dataset["time"],
            dataset["wave_te"],
            ".-",
            label="Energy Period",
            color=dense(0.65),
        )
        ax[1].plot(
            dataset["time"],
            dataset["wave_tz"],
            ".-",
            label="Zero Crossing Period",
            color=dense(0.95),
        )
        ax[1].set(ylim=(0, 22), ylabel="Period [s]")

        ax[2].plot(
            dataset["time"],
            dataset["wave_dp"],
            ".-",
            label="Peak Direction",
            color=haline(0.10),
        )
        ax[2].plot(
            dataset["time"],
            dataset["wave_spread"],
            ".-",
            label="Peak Spread",
            color=haline(0.50),
        )
        ax[2].set(ylabel="Direction [deg]")
        for a in ax:
            a.legend(loc="upper left", bbox_to_anchor=[1.01, 1.0], handlelength=1.5)
        for a in ax[:-1]:
            a.set(xticklabels=[])
        date = dataset.time[0].values.astype(str).split("T")[0]
        ax[0].set(title=f"{dataset.datastream} on {date}")
        ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax[-1].set(xlabel="Time (UTC)")
        plot_file = self.get_ancillary_filepath(title="wave_stats")
        fig.savefig(plot_file)

        ## Wavelet figure
        fig, ax = plt.subplots(
            figsize=(10, 5), subplot_kw={"yscale": "log"}, constrained_layout=True
        )
        vmax = 0.35
        pcm = ax.pcolormesh(
            dataset["time_cwt"].values,
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
            dataset["time_cwt"].values, dataset["frequency"].values
        )
        step_t, step_f = 10, 5  # subsample to avoid a cluttered quiver
        energy = dataset["wavelet_energy_density"].T.values[::step_f, ::step_t]
        energy_thresh = 0.1 * vmax  # only show arrows above 10% of max energy shown
        qu_masked = np.where(
            energy >= energy_thresh, qu.values[::step_f, ::step_t], np.nan
        )
        qv_masked = np.where(
            energy >= energy_thresh, qv.values[::step_f, ::step_t], np.nan
        )
        ax.quiver(
            time_grid[::step_f, ::step_t],
            freq_grid[::step_f, ::step_t],
            qu_masked,
            qv_masked,
            color="black",
            scale=60,
            width=0.002,
        )
        plot_file = self.get_ancillary_filepath(title="wavelet_energy_density")
        fig.savefig(plot_file)

        # Plot directional spectra
        fig, ax = plt.subplots(
            figsize=(8, 6), subplot_kw=dict(projection="polar"), constrained_layout=True
        )
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        # Use frequencies up to 0.5 Hz
        spectrum = dataset["wavelet_dir_energy_density"].mean("time_cwt")
        # Create grid and plot
        a, f = np.meshgrid(np.deg2rad(spectrum["direction"]), 1 / spectrum["frequency"])
        color_level_max = np.nanmax(spectrum.values)
        levels = np.linspace(0, color_level_max, 11)
        c = ax.contourf(a, f, spectrum, levels=levels, cmap="Blues")
        cbar = plt.colorbar(c)
        cbar.set_label(r"ESD [m$^2$/deg]", labelpad=20)
        ax.set_ylim(2, 12)
        ylabels = ax.get_yticklabels()
        ylabels = [ilabel.get_text() for ilabel in ax.get_yticklabels()]
        ylabels = [ilabel + " s" for ilabel in ylabels]
        ticks_loc = ax.get_yticks()
        ax.set_yticks(ticks_loc)
        ax.set_yticklabels(ylabels)
        plot_file = self.get_ancillary_filepath(title="wavelet_directional_spectra")
        fig.savefig(plot_file)

        plt.close("all")
