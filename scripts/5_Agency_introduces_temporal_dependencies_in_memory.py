"""
See manuscript results section: Agency introduces temporal dependencies in memory
"""

import numpy as np, pandas as pd
from scipy.stats import ttest_1samp, f_oneway, ttest_ind, norm
from statsmodels.stats.multitest import multipletests
from pathlib import Path

# ---------- paths ----------
base_data = Path("..") / "data" / "overall_data"
FILES  = {
    "Adventure": base_data / "adventure_result.xlsx",
    "Romance":   base_data / "romance_result.xlsx",
}

CONDS  = ("free","yoke","pasv")
NAMES  = {"free":"Free","yoke":"Yoked","pasv":"Passive"}
pairs  = [("free","yoke"), ("free","pasv")]

pull = lambda df,c,col: df.loc[df.cond.eq(c), col].dropna().astype(float).values
pooled_df = lambda x,y: float(len(x)+len(y)-2)

# ---- Fisher helpers (treat nghb-ef like correlation-like values) ----
fisher = lambda x: np.arctanh(np.clip(np.asarray(x, float), -0.999999, 0.999999))
invf   = lambda z: np.tanh(np.asarray(z, float))

def fisher_one_sample(x):
    """
    One-sample test in Fisher-z space (vs 0).
    Returns: zstat, df, p, mean_raw, mean_back
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
    m, se = np.mean(z), np.std(z, ddof=1) / np.sqrt(n)
    zstat = 0.0 if se == 0 else m / se
    p = 2 * norm.sf(abs(zstat))
    mean_raw = float(np.mean(x))
    mean_back = float(invf(m))
    return float(zstat), float(n - 1), float(p), mean_raw, mean_back

# ---------- 5.1 Neighbor encoding effect ----------
print("5.1.1 The neighbor encoding effect was positive in all three conditions for both stories.\n")
for story, fp in FILES.items():
    df = pd.read_excel(fp, usecols=["cond","nghb-ef"])
    print(story)
    for c in CONDS:
        s = pull(df, c, "nghb-ef")
        n = len(s)
        if n < 2:
            continue

        # Raw one-sample t-test vs 0
        t_raw, p_raw = ttest_1samp(s, 0)
        mean_raw = float(np.mean(s))

        # Fisher-z one-sample
        zstat, df_z, p_z, mean_raw_f, mean_back = fisher_one_sample(s)

        print(f"  {NAMES[c]}:")
        print(f"    Raw stats:        mean={mean_raw:.3f}, t({n-1})={t_raw:.3f}, p={p_raw:.3f}")
        print(f"    Fisher Transform: mean_z={mean_back:.3f}, z({int(df_z)})={zstat:.3f}, p={p_z:.3f}")
    print()

print("5.1.2 The neighbor encoding effect was positive in all three conditions and significantly different across conditions in the Romance story, with Free > Yoked and Free > Passive.\n")
rom = pd.read_excel(FILES["Romance"], usecols=["cond","nghb-ef"])
G   = [pull(rom, c, "nghb-ef") for c in CONDS]
k, n = 3, sum(map(len, G))
F, p = f_oneway(*G)
print(f"Romance ANOVA: F({k-1},{n-k})={F:.2f}, p={p:.3f}")

print("Post-hoc (pooled t-tests, equal variances):")
pooled_ps = []
for a, b in pairs:
    A, B = pull(rom, a, "nghb-ef"), pull(rom, b, "nghb-ef")
    tP, pP = ttest_ind(A, B, equal_var=True)
    dfP = pooled_df(A, B)
    pooled_ps.append(pP)
    print(f"  {NAMES[a]} vs {NAMES[b]}: t({int(dfP)})={tP:.2f}, p={pP:.3f}")

pB = multipletests(pooled_ps, method="bonferroni")[1]
print("  Bonferroni (pooled p): " + ", ".join(
    f"{NAMES[a]} vs {NAMES[b]}: p_bonf={pb:.3f}" for (a, b), pb in zip(pairs, pB)))
print()

# ---------- 5.2 Temporal violation rate ----------
print("5.2 There was no significant difference in temporal violation rate across conditions in either story.\n")
for story, fp in FILES.items():
    df = pd.read_excel(fp, usecols=["cond","tv_rate"])
    G   = [pull(df, c, "tv_rate") for c in CONDS]
    k, n = 3, sum(map(len, G))
    F, p = f_oneway(*G)
    print(f"{story}: F({k-1},{n-k})={F:.2f}, p={p:.3f}")
print()
