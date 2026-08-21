import numpy as np
import xarray as xr
import pycwt
from mhkit import wave, dolfyn

dir_bin_width = 5  # degrees, directional distribution histogram bin width
dir_bins = np.arange(-180, 180 + dir_bin_width, dir_bin_width)
constants = dict(
    fs=2.5,  # Hz, Spotter sampling frequency
    wat=1800,  # s, window averaging time
    wvt_wat=60,  # s wavelet window averaging time
    fft_decimation=6,  # 1/fraction of bin length for FFT vector
    freq_slc=[0.045, 0.5],
    dir_centers=((dir_bins[:-1] + dir_bins[1:]) / 2).astype("float32"),
)


def wave_analysis(ds, wavelet_basic_stats=False, directional_spectra=False):
    # Fill small gaps so we can calculate a wave spectrum
    for key in ["x", "y", "z"]:
        ds[key] = ds[key].interpolate_na(
            dim="time", method="linear", max_gap=np.timedelta64(5, "s")
        )

    # Create 2D tensor for spectral analysis
    disp = xr.DataArray(
        data=np.array(
            [
                ds["x"],
                ds["y"],
                ds["z"],
            ]
        ),
        coords={"dir": ["x", "y", "z"], "time": ds["time"]},
    )

    ## Using dolfyn to create spectra
    nbin = constants["fs"] * constants["wat"]
    fft_tool = dolfyn.adv.api.ADVBinner(
        n_bin=nbin,
        fs=constants["fs"],
        n_fft=nbin // constants["fft_decimation"],
        n_fft_coh=nbin // constants["fft_decimation"],
    )
    # Trim frequency vector to > 0.0455 Hz (wave periods smaller than 22 s)
    slc_freq = slice(constants["freq_slc"][0], constants["freq_slc"][1])

    # Auto-spectra
    psd = fft_tool.power_spectral_density(disp, freq_units="Hz", pct_overlap=0.5)
    psd = psd.sel(freq=slc_freq)
    Sxx = psd.sel(S="Sxx")
    Syy = psd.sel(S="Syy")
    Szz = psd.sel(S="Szz")

    # Cross-spectra
    csd = fft_tool.cross_spectral_density(disp, freq_units="Hz", pct_overlap=0.5)
    csd = csd.sel(coh_freq=slc_freq)
    Cxz = csd.sel(C="Cxz").real
    Cxy = csd.sel(C="Cxy").real
    Cyz = csd.sel(C="Cyz").real

    ## Wave height and period
    pd_Szz = Szz.T.to_pandas()
    Hs = wave.resource.significant_wave_height(pd_Szz)
    Te = wave.resource.energy_period(pd_Szz)
    Ta = wave.resource.average_wave_period(pd_Szz)
    Tp = wave.resource.peak_period(pd_Szz)
    Tz = wave.resource.average_zero_crossing_period(pd_Szz)

    # Check factor: generally should be around 1
    k = np.sqrt((Sxx + Syy) / Szz)

    # Calculate peak wave direction and spread
    a1 = Cxz.values / np.sqrt((Sxx + Syy) * Szz)
    b1 = Cyz.values / np.sqrt((Sxx + Syy) * Szz)
    a2 = (Sxx - Syy) / (Sxx + Syy)
    b2 = 2 * Cxy.values / (Sxx + Syy)
    theta = np.rad2deg(np.arctan2(b1, a1))  # degrees CCW from East, "to" convention
    phi = np.rad2deg(np.sqrt(2 * (1 - np.sqrt(a1**2 + b1**2))))

    # Get peak frequency - fill nan slices with 0
    peak_idx = psd[2].fillna(0).argmax("freq")
    # degrees CW from North ("from" convention)
    direction = (270 - theta[:, peak_idx]) % 360
    # Set direction from -180 to 180
    direction[direction > 180] -= 360
    spread = phi[:, peak_idx]

    # Trim dataset length
    ds_psd = ds.isel(
        time=slice(None, len(psd["time_psd"])),
    )
    # Set time coordinates
    time = xr.DataArray(
        psd["time_psd"].values,
        coords={"time": psd["time_psd"].values},
        attrs=ds["time"].attrs,
    )

    ds_psd = ds_psd.assign_coords({"time": time})
    # Make sure mhkit vars are set to float32
    ds_psd["wave_energy_density"].values = Szz
    ds_psd["wave_hs"].values = Hs.to_xarray().astype("float32")
    ds_psd["wave_te"].values = Te.to_xarray().astype("float32")
    ds_psd["wave_tp"].values = Tp.to_xarray().astype("float32")
    ds_psd["wave_ta"].values = Ta.to_xarray().astype("float32")
    ds_psd["wave_tz"].values = Tz.to_xarray().astype("float32")
    ds_psd["wave_check_factor"].values = k
    ds_psd["wave_dp"].values = direction.astype("float32")
    ds_psd["wave_spread"].values = spread.astype("float32")

    if directional_spectra:
        ds_psd["wave_a1_value"].values = a1
        ds_psd["wave_b1_value"].values = b1
        ds_psd["wave_a2_value"].values = a2
        ds_psd["wave_b2_value"].values = b2
        ## Direct Fourier Transform (DFT) Method for Directional Wave Spectrum
        # Calculate directional wave spectrum
        r1 = np.sqrt(ds_psd["wave_a1_value"] ** 2 + ds_psd["wave_b1_value"] ** 2)
        r2 = np.sqrt(ds_psd["wave_a2_value"] ** 2 + ds_psd["wave_b2_value"] ** 2)
        # dir1 (+/- pi) and dir2 (+/- pi/2) are CCW from East, "to" convention
        dir1 = np.arctan2(ds_psd["wave_b1_value"], ds_psd["wave_a1_value"])
        dir2 = 0.5 * np.arctan2(ds_psd["wave_b2_value"], ds_psd["wave_a2_value"])

        # ds["direction"] is CW from North, "from" convention (shared with the wavelet
        # method) - convert to CCW from East, "to" convention to match dir1/dir2
        # (this transform is self-inverse, so the same formula converts both ways)
        theta = np.deg2rad((270 - ds_psd["direction"]) % 360)

        # Spreading function (parametric estimate of the directional distribution function)
        # Subtract dataset variable to get dimensions right
        spread_func = (1 / np.pi) * (
            0.5 + r1 * np.cos(-1 * (dir1 - theta)) + r2 * np.cos(-2 * (dir2 - theta))
        )

        # Convert D from density per radian to density per degree (direction coord is in degrees)
        # "spread_func" DataArray inherits the original degree convention, so no need to remap it to CW-North
        ds_psd["wave_dir_energy_density"].values = ds_psd["wave_energy_density"] * (
            spread_func * (np.pi / 180)
        )

    ## Wavelets
    w0 = 6  # According to Farge (1992), a commonly used value for the Morlet wavelet.
    mother = pycwt.Morlet(w0)
    freq_target = psd["freq"].values
    Wx_values, _, freq_out, _, _, _ = pycwt.cwt(
        disp[0].values, 1 / constants["fs"], wavelet=mother, freqs=freq_target
    )
    Wy_values, _, _, _, _, _ = pycwt.cwt(
        disp[1].values, 1 / constants["fs"], wavelet=mother, freqs=freq_target
    )
    Wz_values, _, _, _, _, _ = pycwt.cwt(
        disp[2].values, 1 / constants["fs"], wavelet=mother, freqs=freq_target
    )
    Wx = xr.DataArray(
        Wx_values,
        coords={"freq": freq_out, "time": ds["time"].values},
        dims=["freq", "time"],
    )
    Wy = xr.DataArray(
        Wy_values,
        coords={"freq": freq_out, "time": ds["time"].values},
        dims=["freq", "time"],
    )
    Wz = xr.DataArray(
        Wz_values,
        coords={"freq": freq_out, "time": ds["time"].values},
        dims=["freq", "time"],
    )

    # pycwt.cwt implements the exact discrete Torrence & Compo (1998)
    # normalization, so |W|^2 is directly comparable (shape and magnitude)
    # to the Fourier PSD (m^2/Hz)
    Wzz_psd = abs(Wz) ** 2

    if wavelet_basic_stats:
        # Wavelet wave stats
        pd_Wzz_psd = Wzz_psd.to_pandas()
        Hs_cwt = wave.resource.significant_wave_height(pd_Wzz_psd)
        Te_cwt = wave.resource.energy_period(pd_Wzz_psd)
        Ta_cwt = wave.resource.average_wave_period(pd_Wzz_psd)
        Tp_cwt = wave.resource.peak_period(pd_Wzz_psd)
        Tz_cwt = wave.resource.average_zero_crossing_period(pd_Wzz_psd)

    # Estimate wave direction from X and Y wavelet components
    # Cross wavelet transform: magnitude gives cross-wavelet power, angle gives relative phase
    Wyz = Wy * np.conj(Wz)
    Wxz = Wx * np.conj(Wz)
    # Convert wave direction matrix from "CCW from E" to "CW from N"
    direction_cwt = (270 - np.rad2deg(np.arctan2(Wyz.real, Wxz.real))) % 360
    # Set to +/-180 degrees
    direction_cwt = ((direction_cwt + 180) % 360) - 180

    # Find peak wave direction
    peak_idx_ctw = abs(Wzz_psd).fillna(0).argmax("freq")
    Dp_cwt = direction_cwt.isel(freq=peak_idx_ctw)

    # Bin-average wavelet-based parameters to reduce noise
    cwt_tool = dolfyn.adv.api.ADVBinner(
        n_bin=constants["fs"] * constants["wvt_wat"], fs=constants["fs"]
    )
    time_cwt_avg = cwt_tool.mean(ds["time_cwt"].values)
    direction_cwt_avg = cwt_tool.mean(direction_cwt, axis=-1)
    Wzz_psd_avg = cwt_tool.mean(Wzz_psd, axis=-1)
    if wavelet_basic_stats:
        Hs_cwt = cwt_tool.mean(Hs_cwt)
        Te_cwt = cwt_tool.mean(Te_cwt)
        Ta_cwt = cwt_tool.mean(Ta_cwt)
        Tp_cwt = cwt_tool.mean(Tp_cwt)
        Tz_cwt = cwt_tool.mean(Tz_cwt)
        Dp_cwt = cwt_tool.mean(Dp_cwt)

    # Trim dataset length
    ds_out = ds_psd.isel(
        time_cwt=slice(None, len(time_cwt_avg)),
    )
    # Set time coordinates
    time_cwt = xr.DataArray(
        time_cwt_avg,
        coords={"time_cwt": time_cwt_avg},
        attrs=ds["time_cwt"].attrs,
    )
    ds_out = ds_out.assign_coords({"time_cwt": time_cwt})
    ds_out["wave_direction"].values = direction_cwt_avg.T
    ds_out["wavelet_energy_density"].values = Wzz_psd_avg.T
    if wavelet_basic_stats:
        ds_out["wave_hs_cwt"].values = Hs_cwt.astype("float32")
        ds_out["wave_te_cwt"].values = Te_cwt.astype("float32")
        ds_out["wave_tp_cwt"].values = Tp_cwt.astype("float32")
        ds_out["wave_ta_cwt"].values = Ta_cwt.astype("float32")
        ds_out["wave_tz_cwt"].values = Tz_cwt.astype("float32")
        ds_out["wave_dp_cwt"].values = Dp_cwt.astype("float32")

    # Directional distribution function: histogram of direction within each
    # cwt_tool time bin (same windows as Wzz_psd_cwt/D_cwt above), normalized so
    # it integrates to 1 over direction, for each time_cwt/frequency
    # This is an empirical (non-parametric) estimate
    def _hist_per_freq(x, bins):
        x = x[~np.isnan(x)]
        if x.size == 0:
            return np.full(bins.size - 1, np.nan)
        counts, _ = np.histogram(x, bins=bins, density=True)
        return counts

    # Reshape the fast-time direction estimate into cwt_tool's windows: (freq, time_cwt, n_bin)
    dir_samples = cwt_tool.reshape(direction_cwt.values)
    D = np.apply_along_axis(_hist_per_freq, -1, dir_samples, dir_bins)
    # (freq, time_cwt, direction) -> (time_cwt, freq, direction)
    D = np.moveaxis(D, 0, 1)

    # Full 2D wave spectrum: E(time_cwt, f, theta) = S(time_cwt, f) * D(time_cwt, f, theta)
    E = np.asarray(Wzz_psd_avg).T[..., None] * D

    ds_out["directional_distr_func"].values = D
    ds_out["wavelet_dir_energy_density"].values = E

    return ds_out
