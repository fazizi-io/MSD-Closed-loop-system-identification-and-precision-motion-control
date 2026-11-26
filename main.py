import scipy.io
import numpy as np
import polars as pl
import control as ct

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

# Optional: Wrap the time-domain signals into a Polars DataFrame if you want fast vectorized slicing/filtering
# signals_df = pl.DataFrame(signals)
