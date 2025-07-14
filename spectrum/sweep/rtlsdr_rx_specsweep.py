#!/usr/bin/env python3
"""
RTL-SDR Spectrum Sweep using GNU Radio with explicit RTL args.
Dependencies:
  sudo apt-get install gnuradio gr-osmosdr python3-osmosdr
  pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import time

from gnuradio import gr, blocks, fft
import osmosdr
from scipy.signal import decimate

def grab_n_ffts(samp_rate, gain, center_freq, nfft, nframes, rtl_device=0):
    """
    Build & run a mini-flowgraph that:
      RTL-SDR → Stream->Vector → FFT → Head(nframes) → Sink
    Blocks until exactly nframes FFT vectors are captured.
    """
    tb = gr.top_block()

    # NOTE the explicit ordering: numchan=1 first, then rtl=0
    src  = osmosdr.source(args=f"numchan=1 rtl={rtl_device}")
    src.set_sample_rate(samp_rate)
    src.set_center_freq(center_freq)
    src.set_gain(gain)

    window = np.hamming(nfft).tolist()
    stv    = blocks.stream_to_vector(gr.sizeof_gr_complex, nfft)
    fftb   = fft.fft_vcc(nfft, True, window, True)
    head   = blocks.head(gr.sizeof_gr_complex * nfft, nframes)
    snk    = blocks.vector_sink_c(nfft)  # vlen = nfft

    tb.connect(src, stv, fftb, head, snk)
    tb.run()  # blocks until head has passed nframes vectors

    raw = np.array(snk.data(), dtype=np.complex64)
    return raw.reshape(nframes, nfft)


def sweep_band(start_hz, stop_hz, samp_rate, gain, nfft, nframes,
               overlap, dec_factor, fft_hold='avg'):
    step = int(samp_rate * overlap)
    freqs = np.arange(start_hz, stop_hz, step)
    if freqs[-1] + step < stop_hz:
        freqs = np.hstack([freqs, freqs[-1] + step])

    overlap_bins = int(nfft * overlap)
    half_ov      = overlap_bins // 2
    dec_len      = overlap_bins // dec_factor

    sweep = np.zeros((len(freqs), dec_len))

    for i, fc in enumerate(freqs, start=1):
        print(f"[{i}/{len(freqs)}] Tuning to {fc/1e6:.3f} MHz")
        frames = grab_n_ffts(
            samp_rate  = samp_rate,
            gain       = gain,
            center_freq= fc,
            nfft       = nfft,
            nframes    = nframes,
            rtl_device = 0
        )

        # compute power & reorder
        reordered = np.zeros((nframes, overlap_bins))
        for k in range(nframes):
            p   = np.abs(frames[k])**2
            neg = p[nfft//2 + half_ov :]
            pos = p[:half_ov]
            reordered[k] = np.hstack([neg, pos])

        held = reordered.mean(axis=0) if fft_hold=='avg' else reordered.max(axis=0)
        sweep[i-1] = decimate(held, dec_factor, ftype='fir', zero_phase=True)

    return freqs, sweep


def plot_sweep(freqs, sweep, samp_rate, nfft, overlap, dec_factor):
    # stitch frequency axis
    step    = samp_rate * overlap
    f_lo    = freqs[0] - step/2
    f_hi    = freqs[-1] + step/2 - (samp_rate/nfft)
    freq_axis = np.arange(f_lo, f_hi, (samp_rate/nfft)*dec_factor) / 1e6

    data    = sweep.ravel()
    data_dbm= 10*np.log10((data**2)/50 + 1e-30)

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,6), sharex=True)
    ax1.plot(freq_axis, data_dbm)
    ax1.set_ylabel("Power (dBm) [50Ω]")
    ax1.grid(True, ls='--', alpha=0.5)

    ax2.plot(freq_axis, data)
    ax2.set_xlabel("Frequency (MHz)")
    ax2.set_ylabel("Relative Power")
    ax2.grid(True, ls='--', alpha=0.5)

    plt.tight_layout()
    plt.show()


def main():
    # === User parameters ===
    START_FREQ = 25e6         # Hz
    STOP_FREQ  = 1750e6       # Hz
    FS         = 2.8e6        # sample rate in Hz
    GAIN       = 40           # dB
    NFFT       = 4096
    NFRAMES    = 20
    OVERLAP    = 0.5          # 50%
    DEC_FACTOR = 16
    FFT_HOLD   = 'avg'        # 'avg' or 'max'

    freqs, sweep = sweep_band(
        start_hz   = START_FREQ,
        stop_hz    = STOP_FREQ,
        samp_rate  = FS,
        gain       = GAIN,
        nfft       = NFFT,
        nframes    = NFRAMES,
        overlap    = OVERLAP,
        dec_factor = DEC_FACTOR,
        fft_hold   = FFT_HOLD
    )
    plot_sweep(freqs, sweep, FS, NFFT, OVERLAP, DEC_FACTOR)


if __name__ == "__main__":
    main()
