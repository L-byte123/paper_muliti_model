"""Label-free signal features and deterministic diagnostic text (paper Eqs. 2-7)."""
import numpy as np
import pywt


def describe(signal, sample_rate, wavelet="db4", level=2, peak_bins=5):
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1 or len(x) < 8 or not np.isfinite(x).all():
        raise ValueError("Expected finite 1D signal with at least eight samples")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    spectrum = np.fft.rfft(x)
    power = np.abs(spectrum) ** 2
    dominant = int(np.argmax(power))
    total = float(power.sum())
    concentration = float(np.sort(power)[-peak_bins:].sum() / total) if total else 0.0
    # Circular autocorrelation matches the wraparound lags in Fig. 2.
    autocorrelation = np.fft.irfft(spectrum * spectrum.conj(), n=len(x))
    lags = np.argsort(-autocorrelation, kind="stable")[:5].tolist()
    wavelet_obj = pywt.Wavelet(wavelet)
    actual_level = min(level, pywt.dwt_max_level(len(x), wavelet_obj.dec_len))
    coeff = pywt.wavedec(x, wavelet_obj, level=actual_level, mode="symmetric")
    energies = np.array([float(np.square(c).sum()) for c in coeff])
    probabilities = energies / energies.sum() if energies.sum() else energies
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    return dict(minimum=float(x.min()), maximum=float(x.max()), median=float(np.median(x)),
                trend="upward" if x[-1] > x[0] else "downward", lags=lags,
                dominant_bin=dominant, dominant_hz=dominant * sample_rate / len(x),
                spectral_concentration=concentration, wavelet_entropy=entropy,
                wavelet_energies=energies.tolist())


def to_text(signal, sample_rate):
    f = describe(signal, sample_rate)
    # No label, file name or data-dependent class hint is ever included.
    return (f"Input statistics: min value {f['minimum']:.4f}, max value {f['maximum']:.4f}, "
            f"median value {f['median']:.4f}, the trend of input is {f['trend']}, "
            f"top 5 lags are: {f['lags']}, main frequency component f main: "
            f"[{f['dominant_bin']}], wavelet entropy: [{f['wavelet_entropy']:.4f}]. "
            f"Spectral concentration: {f['spectral_concentration']:.4f}.")


def patch_texts(signal, sample_rate, patch_length=32, stride=16, scope="patch"):
    if scope == "window":
        return [to_text(signal, sample_rate)]
    return [to_text(signal[i:i + patch_length], sample_rate)
            for i in range(0, len(signal) - patch_length + 1, stride)]


def add_noise(x, snr_db, rng):
    """Exact per-window SNR in dB; zero-power signals remain zero."""
    power = np.mean(np.square(x), axis=-1, keepdims=True)
    noise = rng.normal(size=x.shape)
    noise /= np.sqrt(np.mean(noise ** 2, axis=-1, keepdims=True)).clip(1e-12)
    return (x + noise * np.sqrt(power / 10 ** (snr_db / 10))).astype(np.float32)
