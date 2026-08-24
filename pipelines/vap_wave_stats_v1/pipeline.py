from typing import Dict
import xarray as xr
from tsdat import TransformationPipeline
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from cmocean.cm import amp_r, dense, haline

from shared.plots import plot_gps


class VapWaveStats(TransformationPipeline):
    """---------------------------------------------------------------------------------
    VAP pipeline for combining Sofar Spotter data products.
    ---------------------------------------------------------------------------------"""

    def hook_customize_input_datasets(self, input_datasets) -> Dict[str, xr.Dataset]:
        # Code hook to customize any input datasets prior to datastreams being combined
        # and data converters being run.

        return input_datasets

    def hook_customize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset before qc is applied
        dataset.attrs.pop("description")

        return dataset

    def hook_finalize_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        # (Optional) Use this hook to modify the dataset after qc is applied
        # but before it gets saved to the storage area
        return dataset

    def hook_plot_dataset(self, dataset: xr.Dataset):
        # (Optional, recommended) Create plots.
        plt.style.use("default")  # clear any styles that were set before

        # Wave stats
        if "air_pressure" in dataset:
            n = 5
        else:
            n = 4

        fig, ax = plt.subplots(n, 1, figsize=(11, 7), constrained_layout=True)
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
        ax[1].set(ylabel="Period [s]")

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

        ax[3].plot(
            dataset["time"],
            dataset["sst"],
            ".-",
            label="Sea Surface Temperature",
            color="black",
        )
        ax[3].set(ylabel="Temperature\n[deg C]")

        if "air_pressure" in dataset:
            ax[4].plot(
                dataset["time"],
                dataset["air_pressure"],
                ".-",
                label="Air Pressure",
                color="black",
            )
            ax[4].set(ylabel="Pressure [hPa]")

        for a in ax:
            a.legend(loc="upper left", bbox_to_anchor=[1.01, 1.0], handlelength=1.5)
        for a in ax[:-1]:
            a.set(xticklabels=[])
        ax[-1].tick_params(labelrotation=45)
        ax[-1].xaxis.set_major_formatter(mdates.DateFormatter("%D %H"))
        ax[-1].set(xlabel="Time (UTC)")

        plot_file = self.get_ancillary_filepath(title="wave_stats")
        fig.savefig(plot_file)

        # Plot GPS
        fig, ax = plot_gps(dataset)
        plot_file = self.get_ancillary_filepath(title="location")
        fig.savefig(plot_file)

        plt.close("all")
