#!/usr/bin/env python3
"""
RTL-SDR Spectrum Sweep (single init, dynamic wait for nfrmhold FFTs)
Dependencies:
  sudo apt-get install gnuradio gr-osmosdr python3-osmosdr
  pip install numpy scipy matplotlib scipy
"""

import time
import numpy as np
import matplotlib.pyplot as plt

from gnuradio import gr, blocks, fft
import osmosdr
from scipy.signal import decimate

class SpectrumSweep(gr.top_block):
    def __init__(self,
                 samp_rate, nfft, nfrmhold,
                 overlap, dec_factor, tuner_gain,
                 rtl_device=0):
        super().__init__()

        # Open RTL stick once
        args = f"numchan=1 rtl={rtl_device}"
        self.src = osmosdr.source(args=args)
        self.src.set_sample_rate(samp_rate)
        self.src.set_gain(tuner_gain)

        # FFT chain: stream→vector→FFT→vector sink (complex)
        window = np.hamming(nfft).tolist()
        self.stv = blocks.stream_to_vector(gr.sizeof_gr_complex, nfft)
        self.fft = fft.fft_vcc(nfft, True, window, True)
        self.snk = blocks.vector_sink_c(nfft)  # collect vectors of length nfft

        self.connect(self.src, self.stv, self.fft, self.snk)

        # store parameters
        self.samp_rate  = samp_rate
        self.nfft       = nfft
        self.nfrmhold   = nfrmhold
        self.overlap    = overlap
        self.dec_factor = dec_factor

    def sweep(self, freqs, fft_hold='avg', flush_secs=0.1):
        """Retune on‐the‐fly and wait until nfrmhold FFTs arrive."""
        self.start()

        overlap_bins = int(self.nfft * self.overlap)
        half_ov      = overlap_bins // 2
        dec_len      = overlap_bins // self.dec_factor
        sweep_matrix = np.zeros((len(freqs), dec_len))

        for i, fc in enumerate(freqs):
            print(f"[{i+1}/{len(freqs)}] Tuning to {fc/1e6:.3f} MHz")
            self.src.set_center_freq(fc)
            time.sleep(flush_secs)       # flush old samples
            self.snk.reset()             # clear sink

            # wait until we have at least nfrmhold*nfft samples
            needed = self.nfrmhold * self.nfft
            timeout = needed/self.samp_rate * 3  # generous timeout
            t0 = time.time()
            while True:
                raw = np.array(self.snk.data(), dtype=np.complex64)
                if raw.size >= needed:
                    break
                if time.time() - t0 > timeout:
                    raise RuntimeError(f"Timeout waiting for {needed} samples at {fc/1e6} MHz")
                # sleep roughly one frame's worth
                time.sleep(self.nfft / self.samp_rate)

            # only keep exactly needed samples
            raw = raw[:needed]
            frames = raw.reshape(self.nfrmhold, self.nfft)

            # compute power and extract overlap window
            reordered = np.zeros((self.nfrmhold, overlap_bins))
            for k in range(self.nfrmhold):
                p    = np.abs(frames[k])**2
                neg  = p[self.nfft//2 + half_ov:]
                pos  = p[:half_ov]
                reordered[k] = np.hstack([neg, pos])

            # avg or max hold
            held = reordered.mean(axis=0) if fft_hold=='avg' else reordered.max(axis=0)

            # FIR‐decimate to smooth
            sweep_matrix[i] = decimate(held, self.dec_factor, ftype='fir', zero_phase=True)

        self.stop()
        self.wait()
        return sweep_matrix

def main():
    # parameters
    START_HZ   = 25e6
    STOP_HZ    = 1750e6
    FS         = 2.8e6
    GAIN       = 40
    NFFT       = 4096
    NFRAMES    = 20
    OVERLAP    = 0.5
    DEC_FACTOR = 16
    FFT_HOLD   = 'avg'

    # build freq hops
    step  = FS * OVERLAP
    freqs = np.arange(START_HZ, STOP_HZ, step)
    if freqs[-1] + step < STOP_HZ:
        freqs = np.hstack([freqs, freqs[-1] + step])

    tb = SpectrumSweep(FS, NFFT, NFRAMES, OVERLAP, DEC_FACTOR, GAIN)
    print(f"Sweeping {START_HZ/1e6:.0f}→{STOP_HZ/1e6:.0f} MHz in {len(freqs)} steps…")
    t0 = time.time()
    sweep = tb.sweep(freqs, fft_hold=FFT_HOLD, flush_secs=0.1)
    print(f"Done in {time.time()-t0:.1f}s")

    # stitch & plot
    bin_w    = FS / NFFT
    f_lo     = freqs[0] - (FS*OVERLAP)/2
    f_hi     = freqs[-1] + (FS*OVERLAP)/2 - bin_w
    freq_axis= np.arange(f_lo, f_hi, bin_w*DEC_FACTOR)/1e6

    data     = sweep.ravel()
    data_dbm = 10*np.log10((data**2)/50 + 1e-30)

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,6), sharex=True)
    ax1.plot(freq_axis, data_dbm, lw=1.2)
    ax1.set_ylabel("Power (dBm) [50Ω]"); ax1.grid(True, ls='--', alpha=0.5)
    ax2.plot(freq_axis, data, lw=1.2)
    ax2.set_xlabel("Frequency (MHz)"); ax2.set_ylabel("Relative Power"); ax2.grid(True, ls='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("spectrum_sweep_USA_LA_USC_BHE112_20250713.png", dpi=900)
    plt.show()

if __name__ == "__main__":
    main()
