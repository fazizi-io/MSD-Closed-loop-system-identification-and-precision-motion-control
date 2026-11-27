import scipy.io
import numpy as np
import polars as pl
import control as ct
import matplotlib.pyplot as plt
from scipy import signal

# 1. Load Plant Transfer Function from CSV using Polars
df = pl.read_csv("MSD2026_Plant_Coefficients.csv").sort("s_power", descending=True)

num = df["numerator"].to_numpy()
den = df["denominator"].to_numpy()
delay = df["input_delay_s"][0]
dt = df["sample_time_s"][0]  # 0 -> continuous-time

P_nodelay = ct.tf(num, den, dt)
print("=== Plant Transfer Function (without delay) ===")
print(P_nodelay)
print(f"Input delay: {delay} s")

# Note: In frequency domain (e.g., FRF / Bode / Nyquist), apply the exact delay via:
# P_jw = ct.frequency_response(P_nodelay, omega).fresp[0, 0, :] * np.exp(-1j * omega * delay)
# Or approximate in time-domain using Padé:
# num_d, den_d = ct.pade(delay, n=4)
# P_pade = P_nodelay * ct.tf(num_d, den_d)

# 2. Load Closed-Loop SysID Signals from .mat
raw_signals = scipy.io.loadmat(
    "MSD2026_Signals_for_CLSysID.mat", squeeze_me=True, simplify_cells=True
)
signals = {k: v for k, v in raw_signals.items() if not k.startswith("__")}

print("\n=== MSD2026_Signals_for_CLSysID.mat ===")
for k, v in signals.items():
    print(f"  {k} (array) -> shape: {v.shape}, dtype: {v.dtype}")

## 3. Setup Time Vector & Sampling Rate
# Since dt in the CSV is 0 (continuous plant), define the data acquisition sampling rate.
# Note: Check your assignment sheet for exact Fs (commonly 1000 Hz or 2000 Hz -> ~507s or ~253s of data).
Fs = 1000.0  # [Hz] Adjust if your setup runs at a different rate
Ts = 1.0 / Fs

r = signals["r"]  # Reference / Excitation
y = signals["y"]  # Measured Output
u = signals["u"]  # Control Input
e = signals["e"]  # Tracking Error (check convention: e = r - y or r - y - d)

N = len(y)
t = np.arange(N) * Ts

# Quick sanity check on loop topology (verify whether excitation enters at r or as a disturbance d)
print("\n=== Signal Diagnostics ===")
print(f"Max |e - (r - y)|: {np.max(np.abs(e - (r - y))):.4e}")
print(f"RMS values -> r: {np.std(r):.4f}, y: {np.std(y):.4f}, u: {np.std(u):.4f}, e: {np.std(e):.4f}")

# 4. Time-Domain Plots (Full & Zoomed Window)
# Downsample slightly for fast rendering if plotting 500k+ points, or plot a representative window
step = 10  # Plot every 10th sample for the full overview to keep matplotlib responsive

fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
fig.suptitle("Closed-Loop SysID Time-Domain Signals (Downsampled Overview)", fontsize=13)

axes[0].plot(t[::step], r[::step], lw=0.8, color="tab:blue")
axes[0].set_ylabel("Reference r")

axes[1].plot(t[::step], y[::step], lw=0.8, color="tab:orange")
axes[1].set_ylabel("Output y")

axes[2].plot(t[::step], u[::step], lw=0.8, color="tab:green")
axes[2].set_ylabel("Control u")

axes[3].plot(t[::step], e[::step], lw=0.8, color="tab:red")
axes[3].set_ylabel("Error e")
axes[3].set_xlabel("Time [s]")

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()

# 5. Frequency-Domain Plots: Power Spectral Density (Welch) & Empirical Plant FRF
# Using Welch's method with Hann window and 50% overlap to inspect excitation bandwidth & resonances
nperseg = 2**13  # 8192 samples -> ~0.12 Hz frequency resolution at 1000 Hz
f_psd, P_rr = signal.welch(r, fs=Fs, window="hann", nperseg=nperseg)
_, P_yy = signal.welch(y, fs=Fs, window="hann", nperseg=nperseg)
_, P_uu = signal.welch(u, fs=Fs, window="hann", nperseg=nperseg)
_, P_ee = signal.welch(e, fs=Fs, window="hann", nperseg=nperseg)

# Cross-spectral densities for Indirect Closed-Loop FRF: P_hat(jw) = S_yr(jw) / S_ur(jw)
_, S_yr = signal.csd(y, r, fs=Fs, window="hann", nperseg=nperseg)
_, S_ur = signal.csd(u, r, fs=Fs, window="hann", nperseg=nperseg)
_, coh_ur = signal.coherence(u, r, fs=Fs, window="hann", nperseg=nperseg)
_, coh_yr = signal.coherence(y, r, fs=Fs, window="hann", nperseg=nperseg)

P_emp = S_yr / S_ur  # Unbiased empirical plant estimate (assuming r is uncorrelated with noise)

# Parametric Plant FRF (with exact delay) on the same frequency grid (excluding DC = 0 Hz)
valid = f_psd > 0
f_hz = f_psd[valid]
omega = 2 * np.pi * f_hz
P_model = ct.frequency_response(P_nodelay, omega).fresp[0, 0, :] * np.exp(-1j * omega * delay)

fig_freq, axes_f = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fig_freq.suptitle("Spectral Content & Initial Empirical vs. Parametric Plant FRF", fontsize=13)

# Subplot 1: PSDs
axes_f[0].loglog(f_hz, P_rr[valid], label="PSD(r)", alpha=0.8)
axes_f[0].loglog(f_hz, P_uu[valid], label="PSD(u)", alpha=0.8)
axes_f[0].loglog(f_hz, P_yy[valid], label="PSD(y)", alpha=0.8)
axes_f[0].set_ylabel("PSD [units²/Hz]")
axes_f[0].legend(loc="best")
axes_f[0].grid(True, which="both", alpha=0.3)

# Subplot 2: Bode Magnitude (Empirical vs Given Model)
axes_f[1].semilogx(f_hz, 20 * np.log10(np.abs(P_emp[valid])), label="Empirical P_hat = S_yr / S_ur", alpha=0.75)
axes_f[1].semilogx(f_hz, 20 * np.log10(np.abs(P_model)), label="Given CSV Plant (with delay)", lw=1.8, linestyle="--", color="black")
axes_f[1].set_ylabel("Magnitude [dB]")
axes_f[1].legend(loc="best")
axes_f[1].grid(True, which="both", alpha=0.3)

# Subplot 3: Coherence (shows which frequency bands have high SNR)
axes_f[2].semilogx(f_hz, coh_ur[valid], label="Coherence(u, r)")
axes_f[2].semilogx(f_hz, coh_yr[valid], label="Coherence(y, r)")
axes_f[2].set_ylabel("Coherence γ²")
axes_f[2].set_xlabel("Frequency [Hz]")
axes_f[2].set_ylim([0, 1.05])
axes_f[2].legend(loc="best")
axes_f[2].grid(True, which="both", alpha=0.3)

plt.tight_layout()
plt.show()