# VAP Fourier vs Wavelets Transformation Pipeline

This VAP pipeline is used to compare the results of FFT/Welch-based spectral analysis vs Wavelet analysis.
It is for documentation purposes.

## Complex Wavelets

**Magnitude (`abs(Wz)`)**
This is the local amplitude of the wave oscillation at each `(freq, time)` point, in meters. It's directly analogous to a smoothly-varying amplitude estimate of a bandpass-filtered version of `z` centered on that frequency. When you square it (`abs(Wz)**2`), you get something proportional to energy/power density in $m^2$ — this is why it's used as `wavelet_power_density` and compared against the Fourier PSD (`Szz`, in $m^2/Hz$). Note pywavelets' CWT isn't automatically Parseval-normalized the way an FFT-based PSD is, so `abs(Wzz)**2` is only proportional to (not exactly equal to) the same units as `Szz` unless you apply the appropriate wavelet normalization factor.

**Angle (`np.angle(Wz)`)**
This is the instantaneous phase of the oscillatory component of `z` at that scale/time, in radians ($-\pi$ to $\pi$). On its own, the phase of a single signal isn't very meaningful physically — it just tracks where in its oscillation cycle the wave is (crest, trough, zero-crossing, etc.) at that instant. It becomes meaningful when compared *between* two signals (like your `Wxz`/`Wyz` cross-wavelet transforms), where the phase difference indicates the relative timing/lag between horizontal and vertical displacement — which is what lets you back out wave direction, just as `Cxz`/`Cyz` phase does for the Fourier cross-spectra.

## Fourier vs Wavelet Directional Distribution Function

They are the same physical quantity, just estimated two different ways. Both `spreading_func` and `directional_distr_func` are estimates of the same thing: the normalized **directional distribution function** $D(f,\theta)$, defined so that:

$$\int_0^{2\pi} D(f,\theta)\,d\theta = 1 \quad \text{for every frequency}$$

and used identically to decompose the 1D frequency spectrum $S(f)$ into a full 2D directional spectrum:

$$E(f,\theta) = S(f)\cdot D(f,\theta)$$

- `spreading_func` is the **parametric** estimate of $D(f,\theta)$ — reconstructed from just the first two directional Fourier moments ($a_1,b_1,a_2,b_2$), giving a fixed 2-harmonic cosine shape.
- `directional_distr_func` (from the histogram) is the **non-parametric/empirical** estimate of the same $D(f,\theta)$ — built directly from many instantaneous cross-wavelet phase (direction) samples.

So `wave_dir_energy_density` and `wavelet_dir_energy_density` are two independent estimates of the *same* underlying 2D directional wave spectrum $E(f,\theta)$ — which is exactly why it made sense earlier to put them on the same `direction` coordinate convention (CW-from-North) for direct, apples-to-apples comparison. Any differences between them reflect estimator bias/variance (2-harmonic truncation vs. histogram sampling noise), not a difference in what physical parameter they represent.


## Two fundamentally different ways of building $D(f,\theta)$

Both pipelines use the same underlying physical information — the cross-spectra between horizontal position (x, y) and vertical elevation (z) — but they extract the directional distribution in very different ways.

### Fourier method
```python
D = (1/π) * (0.5 + r1*cos(θ - dir1) + r2*cos(2*(θ - dir2)))
```
This is the classic **truncated Fourier series** (Longuet-Higgins/Kuik 1988) approach, using only the first two directional moments (`a1,b1,a2,b2`) computed once per burst/frequency from the Welch-averaged cross-spectra (`Cxz`, `Cyz`, `Cxy`).
- **Parametric**: it *assumes* the true $D(f,\theta)$ can be approximated by a cosine series truncated at 2 harmonics.
- **Smooth by construction** — one direction/one spread value drives the whole curve at each frequency.
- Can produce small **negative** "energy" values (a known artifact of truncating the series), which usually get clipped.
- Structurally **cannot resolve multi-modal seas** (e.g. swell arriving from one direction + local wind-sea from another) — it just blurs them into a single bump or a shallow bimodal shape.
- Its "sample size" is really the number of FFT segments averaged into `Cxz`/`Cyz` (Welch averaging, `pct_overlap=0.5`), so variance is already baked in before `directional_distr_func` is built.

### Wavelet-histogram method
```python
direction_cwt_freq = phase of Wxz, Wyz   # one direction estimate per (time, frequency)
D = histogram of direction_cwt_freq over time, per frequency, density=True
```
This directly follows the paper: instead of ensemble-averaging cross-spectra first, you get **one direction estimate per individual wavelet coefficient in time**, then build $D(f,\theta)$ empirically as a **histogram of many such instantaneous estimates** over the ~30 min record.
- **Non-parametric**: no assumed functional shape — in principle it can represent arbitrarily multi-modal, narrow, or skewed distributions.
- Each individual instantaneous direction estimate is very noisy (essentially a 1-degree-of-freedom estimate, like an unaveraged periodogram), but the noise is fought by **quantity of independent-ish time samples** rather than spectral averaging.
- **Always non-negative** by construction (it's literally a count).
- Resolution/quality is governed by your bin width (5°) vs. how many effective independent time samples fall in each `time_cwt` bin — too few samples and bins get noisy/sparse; too fine a bin width and you need more samples to resolve it.

### Practical implications
| | Fourier (`a1,b1,a2,b2`) | Wavelet histogram |
|---|---|---|
| Shape flexibility | Fixed 2-harmonic cosine shape | Free-form, can be multi-modal |
| Can go negative | Yes (needs clipping) | No |
| Smoothness | Very smooth (deterministic given a1/b1/a2/b2) | Depends on # of samples/bin width — can be noisy |
| What it captures | Mean direction + spread (1st/2nd moments only) | Full empirical shape (if enough data) |
| Where averaging happens | Before `directional_distr_func` (FFT segment averaging) | After `directional_distr_func` (histogram/binning over time) |

If you want a sanity check between the two, you could compare them directly for the same burst: take the time-average of your wavelet-derived `wavelet_dir_energy_density` and overlay it against the Fourier-derived `wave_dir_energy_density` for the same frequency range — they should agree reasonably well for simple (unimodal) sea states, and diverge specifically where the Fourier method's 2-harmonic truncation can't represent a bimodal/multi-peaked sea that the histogram approach can.

## Welch vs Wavelet Energy Density

- `wave_energy_density` (`Szz`) is the **Welch/FFT-based** power spectral density estimate of $S(f)$ — a non-parametric, empirical estimate built by averaging periodograms across overlapping FFT segments (`fft_tool.power_spectral_density(..., pct_overlap=0.5)`).
- `wavelet_energy_density` (`Wzz_psd_cwt`) is **also** a non-parametric, empirical estimate of that *same* physical quantity $S(f)$ — just computed via a different transform (`|W_z|^2` from the continuous wavelet transform), then bin-averaged over each `cwt_tool` window instead of FFT-segment-averaged.

Both are estimating the same thing (elevation variance spectral density, m²/Hz) — neither is a parametric model.

Before switching libraries, `pywt.cwt` (PyWavelets) produced raw wavelet coefficients with an undocumented/inconsistent normalization — squaring those gave something with units of `m²` (a wavelet *power*, not a *density*), which is why `Hs_cwt`/`Te_cwt`/etc. computed from it were wrong (`mhkit.wave.resource.*` functions expect a proper PSD in `m²/Hz`).

Switching to `pycwt.cwt` (a direct port of the exact discrete Torrence & Compo 1998 normalization) fixed this: `|W_z|²` from `pycwt` comes out directly comparable in shape *and* magnitude to the Fourier PSD `Szz` — i.e. genuinely `m²/Hz` — because T&C's discrete normalization accounts for the wavelet scale spacing (`dj`) and the mother wavelet's energy (`Cδ` for the Morlet), converting the raw transform coefficients into a proper spectral density rather than an arbitrary-unit coefficient magnitude.


## Prerequisites

* Ensure that your development environment has been set up according to
[the instructions](../../README.md#development-environment-setup).

## Running your pipeline

1. Navigate to the repository root from the terminal (i.e., 2 levels up from this file)
2. Run `runner.py` and specify the transformation pipeline that should run:

        ```shell
        python runner.py vap pipelines/vap_fourier_vs_wavelet/config/pipeline.yaml -b 20230324 -e 20230325
        ```


## Testing your pipeline
This template is set up with a pytest unit test to ensure your pipeline is working correctly.  It is intended that the
pytest unit tests will be run automatically before pipeline deployment to prevent against breaking code changes.  To
run your tests locally, run these commands from your anaconda environment shell:

```bash
cd $REPOSITORY_ROOT
pytest
```
