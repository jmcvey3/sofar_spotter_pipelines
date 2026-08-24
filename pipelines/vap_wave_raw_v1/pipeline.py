import numpy as np
import xarray as xr
from typing import Dict
from tsdat import TransformationPipeline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cmocean.cm import amp_r, dense, haline

from shared.wave_analysis import constants, wave_analysis
from shared.plots import wave_spectra, wave_rose, wavelets, directional_spectra


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
        # Create directional spectra from Welch-PSD and wavelet-determined directions
        ds["wave_dir_energy_density"].values = (
            ds["wave_energy_density"] * ds["directional_distr_func"]
        )

        # Remove variables not useful for users
        return ds.drop_vars(("x", "y", "z", "directional_distr_func", "spreading_func"))

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset after qc is applied
        # but before it gets saved to the storage area
        return dataset

    def hook_plot_dataset(self, dataset: xr.Dataset):
        # (Optional, recommended) Create plots.
        plt.style.use("default")  # clear any styles that were set before

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
            dataset["wave_dm"],
            ".-",
            label="Mean Direction",
            color=haline(0.30),
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
            color=haline(0.70),
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

        # Wave spectra figure
        fig, ax = wave_spectra(dataset)
        plot_file = self.get_ancillary_filepath(title="elevation_spectrum")
        fig.savefig(plot_file)

        # Plot wave rose
        fig, ax = wave_rose(dataset)
        plot_file = self.get_ancillary_filepath(title="wave_rose")
        fig.savefig(plot_file)

        # Plot wavelets and directions
        fig, ax = wavelets(dataset)
        plot_file = self.get_ancillary_filepath(title="wavelet_energy_density")
        fig.savefig(plot_file)

        # Plot directional spectra
        fig, ax = directional_spectra(dataset["wave_dir_energy_density"].mean("time"))
        plot_file = self.get_ancillary_filepath(title="directional_spectra")
        fig.savefig(plot_file)

        plt.close("all")
