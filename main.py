import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import control as ct

files = [
    "MSD2026_Plant_Coefficients.csv",
    "MSD2026_Plant_tf.mat",
    "MSD2026_Signals_for_CLSysID.mat",
]

loaded = {}

for fname in files:
    # simplify_cells=True automatically converts MATLAB structs/cells into clean Python dicts/lists
    raw = scipy.io.loadmat(fname, squeeze_me=True, simplify_cells=True)
    
    # Strip internal MATLAB header keys
    clean = {k: v for k, v in raw.items() if not k.startswith("__")}
    loaded[fname] = clean

    print(f"\n=== {fname} ===")
    for k, v in clean.items():
        if isinstance(v, dict):
            print(f"  {k} (dict/struct) -> keys: {list(v.keys())}")
        elif isinstance(v, np.ndarray):
            print(f"  {k} (array) -> shape: {v.shape}, dtype: {v.dtype}")
        else:
            print(f"  {k} ({type(v).__name__}) -> {v}")