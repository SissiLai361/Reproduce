"""
See manuscript results section: Event recall was predicted by semantic and causal centrality
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, norm
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

def mean_raw_and_backtransformed(x):
    """Return mean in raw space and mean after Fisher-z back-transform."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan
    m_raw = float(np.mean(x))
    m_back = float(invf(np.mean(fisher(x))))
    return m_raw, m_back

def fisher_one_sample(x):
    """
    One-sample test in Fisher-z space (vs 0).
    Returns z-statistic, df, p-value, raw mean, back-transformed mean.
    """
    z = fisher(x)
    z = z[np.isfinite(z)]
    n = z.size
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    m = np.mean(z)
    se = np.std(z, ddof=1) / np.sqrt(n)
    zstat = 0.0 if se == 0 else m / se
    p = 2 * norm.sf(abs(zstat))

    mean_raw, mean_back = mean_raw_and_backtransformed(x)
    return float(zstat), float(n - 1), float(p), mean_raw, mean_back

# ============================================================
# 3.1 Semantic centrality effects
# ============================================================

print("3.1 For both stories, semantic centrality significantly predicted recall (one-sample tests against zero).\n")

for story, fp in FILES.items():
    df = pd.read_excel(fp, usecols=["cond", "idv-sem-ef"])
    print(f"{story} — Semantic")

    for c in CONDS:
        s = pull(df, c, "idv-sem-ef")
        n = len(s)

        # Raw-space test
        t_raw, p_raw = ttest_1samp(s, 0)
        mean_raw = float(np.mean(s)) if n > 0 else np.nan

        # Fisher z test
        zstat, df_z, p_z, m_raw_f, m_back = fisher_one_sample(s)

        print(f"  {NAME[c]}:")
        print(f"    Raw stats:        mean={mean_raw:.3f}, t({n-1})={t_raw:.3f}, p={p_raw:.3f}")
        print(f"    Fisher Transform: mean_z={m_back:.3f}, z({int(df_z)})={zstat:.3f}, p={p_z:.3f}")

    print()

# ============================================================
# 3.2 Causal centrality effects
# ============================================================

print("3.2 For both stories, causal centrality significantly predicted recall (one-sample tests against zero).\n")

for story, fp in FILES.items():
    df = pd.read_excel(fp, usecols=["cond", "idv-caus-ef"])
    print(f"{story} — Causal")

    for c in CONDS:
        s = pull(df, c, "idv-caus-ef")
        n = len(s)

        # Raw-space test
        t_raw, p_raw = ttest_1samp(s, 0)
        mean_raw = float(np.mean(s)) if n > 0 else np.nan

        # Fisher z test
        zstat, df_z, p_z, m_raw_f, m_back = fisher_one_sample(s)

        print(f"  {NAME[c]}:")
        print(f"    Raw stats:        mean={mean_raw:.3f}, t({n-1})={t_raw:.3f}, p={p_raw:.3f}")
        print(f"    Fisher Transform: mean_z={m_back:.3f}, z({int(df_z)})={zstat:.3f}, p={p_z:.3f}")

    print()
