"""
See manuscript results section: Event recall was predicted by semantic and causal centrality
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from pathlib import Path

# ---------- Load paths ----------
base_data = Path("..") / "data" / "overall_data"
FILES = {
    "Adventure": base_data / "adventure_result.xlsx",
    "Romance":   base_data / "romance_result.xlsx",
}

# Conditions and pretty names
CONDS = ("free", "yoke", "pasv")
NAME  = {"free": "Free", "yoke": "Yoked", "pasv": "Passive"}

# Helper to extract numeric vectors
pull = lambda df, cond, col: df.loc[df.cond.eq(cond), col].dropna().astype(float).values

# ---------- Fisher transform helpers ----------
fisher = lambda x: np.arctanh(np.clip(np.asarray(x, float), -0.999999, 0.999999))
invf   = lambda z: np.tanh(np.asarray(z, float))

def fisher_one_sample(x):
    """
    One-sample t-test in Fisher-z space (vs 0), analogous to correlation tests.
    Returns:
      mean_z       - mean Fisher z
      mean_back    - tanh(mean_z), back-transformed mean r
      t_z          - t-statistic in z space
      df           - degrees of freedom
      p_z          - p-value from t-test
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    z = fisher(x)
    z = z[np.isfinite(z)]
    n = z.size
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    # t-test on z vs 0
    t_z, p_z = ttest_1samp(z, 0.0)

    mean_z   = float(np.mean(z))
    mean_back = float(invf(mean_z))
    df       = n - 1

    return mean_z, mean_back, float(t_z), float(df), float(p_z)

# ============================================================
# 3.1 Semantic centrality effects
# ============================================================

print("3.1 For both stories, semantic centrality significantly predicted recall (one-sample tests against zero).\n")

for story, fp in FILES.items():
    # now use sem-ef instead of idv-sem-ef
    df = pd.read_excel(fp, usecols=["cond", "sem-ef"])
    print(f"{story} — Semantic")

    for c in CONDS:
        s = pull(df, c, "sem-ef")
        n = len(s)

        if n < 2:
            continue

        # Raw-space test
        t_raw, p_raw = ttest_1samp(s, 0)
        mean_raw = float(np.mean(s))

        # Fisher z t-test
        mean_z, mean_back, t_z, df_z, p_z = fisher_one_sample(s)

        print(f"  {NAME[c]}:")
        print(f"    Raw stats:        mean={mean_raw:.3f}, t({n-1})={t_raw:.3f}, p={p_raw:.3f}")
        print(f"    Fisher Transform: mean_z={mean_z:.3f}, t({int(df_z)})={t_z:.3f}, p={p_z:.3f}")
        # If you want to also see the back-transformed mean r:
        # print(f"    (back-transformed mean r = {mean_back:.3f})")

    print()

# ============================================================
# 3.2 Causal centrality effects
# ============================================================

print("3.2 For both stories, causal centrality significantly predicted recall (one-sample tests against zero).\n")

for story, fp in FILES.items():
    # now use caus-ef instead of idv-caus-ef
    df = pd.read_excel(fp, usecols=["cond", "caus-ef"])
    print(f"{story} — Causal")

    for c in CONDS:
        s = pull(df, c, "caus-ef")
        n = len(s)

        if n < 2:
            continue

        # Raw-space test
        t_raw, p_raw = ttest_1samp(s, 0)
        mean_raw = float(np.mean(s))

        # Fisher z t-test
        mean_z, mean_back, t_z, df_z, p_z = fisher_one_sample(s)

        print(f"  {NAME[c]}:")
        print(f"    Raw stats:        mean={mean_raw:.3f}, t({n-1})={t_raw:.3f}, p={p_raw:.3f}")
        print(f"    Fisher Transform: mean_z={mean_z:.3f}, t({int(df_z)})={t_z:.3f}, p={p_z:.3f}")
        # print(f"    (back-transformed mean r = {mean_back:.3f})")

    print()
