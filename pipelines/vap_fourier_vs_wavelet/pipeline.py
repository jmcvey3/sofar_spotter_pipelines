import numpy as np
import xarray as xr
from typing import Dict
from tsdat import TransformationPipeline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cmocean.cm import amp_r, dense, haline

from shared.wave_analysis import constants, wave_analysis
from shared.plots import wave_spectra, directional_spectra, wavelets


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

        # Conduct comparison analysis between Welch-PSD and Morlet Wavelet estimates
        ds = wave_analysis(dataset, wavelet_basic_stats=True, directional_spectra=True)
        # Create directional spectra
        ds["wave_dir_energy_density"].values = (
            ds["wave_energy_density"] * ds["spreading_func"]
        )
        ds["wavelet_dir_energy_density"].values = (
            ds["wavelet_energy_density"] * ds["directional_distr_func"]
        )

        return ds.drop_vars(("x", "y", "z"))

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset after qc is applied
        # but before it gets saved to the storage area
        return dataset

    def hook_plot_dataset(self, dataset: xr.Dataset):
        # (Optional, recommended) Create plots.
        plt.style.use("default")  # clear any styles that were set before

        # Wave spectra figure
        fig, ax = wave_spectra(dataset)
        plot_file = self.get_ancillary_filepath(title="elevation_spectra")
        fig.savefig(plot_file)

        # Comparison between mean PSD and wavelet spectra
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.subplots_adjust(left=0.14, right=0.95, top=0.95, bottom=0.1)
        ax.loglog(
            dataset["frequency"],
            dataset["wave_energy_density"].mean("time"),
            label="Welch PSD",
        )
        ax.loglog(
            dataset["frequency"],
            dataset["wavelet_energy_density"].mean("time"),
            label="Morlet Wavelet",
        )
        m = -4
        x = np.logspace(-1, 0)
        y = 10 ** (-4) * x**m
        ax.loglog(x, y, "--", c="black", label="f^-4")
        ax.set(
            ylim=(0.0005, 1),
            xlabel="Frequency [Hz]",
            ylabel="Energy Density [m^2/Hz]",
        )
        ax.grid()
        ax.legend()
        plot_file = self.get_ancillary_filepath(title="mean_spectrum_comparison")
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
        ax[0].plot(
            dataset["time"],
            dataset["wave_hs_cwt"],
            "+",
            label="Significant Wave Height (wavelet)",
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
            dataset["wave_ta_cwt"],
            "+",
            label="Mean Period (wavelet)",
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
            dataset["wave_tp_cwt"],
            "+",
            label="Peak Period (wavelet)",
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
            dataset["wave_te_cwt"],
            "+",
            label="Energy Period (wavelet)",
            color=dense(0.65),
        )
        ax[1].plot(
            dataset["time"],
            dataset["wave_tz"],
            ".-",
            label="Zero Crossing Period",
            color=dense(0.95),
        )
        ax[1].plot(
            dataset["time"],
            dataset["wave_tz_cwt"],
            "+",
            label="Zero Crossing Period (wavelet)",
            color=dense(0.95),
        )
        ax[1].set(ylim=(0, 22), ylabel="Period [s]")

        ax[2].plot(
            dataset["time"],
            dataset["wave_dm"],
            ".-",
            label="Mean Direction",
            color=haline(0.30),
        )
        ax[2].plot(
            dataset["time"],
            dataset["wave_dp"],
            ".-",
            label="Peak Direction",
            color=haline(0.10),
        )
        ax[2].plot(
            dataset["time"],
            dataset["wave_dp_cwt"],
            "+",
            label="Peak Direction (wavelet)",
            color=haline(0.10),
        )
        ax[2].plot(
            dataset["time"],
            dataset["wave_sp"],
            ".-",
            label="Peak Spread",
            color=haline(0.50),
        )
        ax[2].plot(
            dataset["time"],
            dataset["wave_sm"],
            ".-",
            label="Mean Spread",
            color=haline(0.7),
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
        fig, ax = wavelets(dataset)
        plot_file = self.get_ancillary_filepath(title="wavelet_energy_density")
        fig.savefig(plot_file)

        # Plot wavelet directional spectra
        fig, ax = directional_spectra(
            dataset["wavelet_dir_energy_density"].mean("time")
        )
        plot_file = self.get_ancillary_filepath(title="wavelet_directional_spectra")
        fig.savefig(plot_file)

        # Plot Fourier directional spectra
        fig, ax = directional_spectra(dataset["wave_dir_energy_density"].mean("time"))
        plot_file = self.get_ancillary_filepath(title="directional_spectra")
        fig.savefig(plot_file)

        # Use PSD with wavelet directions
        dir_energy_density = (
            dataset["wave_energy_density"] * dataset["directional_distr_func"]
        )
        fig, ax = directional_spectra(dir_energy_density.mean("time"))
        plot_file = self.get_ancillary_filepath(title="fft-wavelet_directional_spectra")
        fig.savefig(plot_file)

        plt.close("all")
