"""
See manuscript results section: Agency magnified individual variability in recall and choice
"""

import numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind, norm, t as tdist
from statsmodels.stats.oneway import anova_oneway  # one-way ANOVA (equal variances)
from statsmodels.stats.multitest import multipletests

# ------------------------ Load data ------------------------
base = Path("..") / "data" / "overall_data"
XLSX = base / "romance_result.xlsx"

ISC  = pd.read_excel(XLSX, sheet_name="merge_isc_by-cond")
pull = lambda cond, col: (
    ISC.loc[ISC["cond"].eq(cond), col].dropna().astype(float).values
)
LABEL = {"free": "Free", "yoke": "Yoked", "pasv": "Passive"}

# ------------------------ Fisher helpers ------------------------
fisher = lambda r: np.arctanh(
    np.clip(np.asarray(r, float), -0.999999, 0.999999)
)
invf   = lambda z: np.tanh(np.asarray(z, float))

def mean_rs(r):
    """Return mean r (raw) and mean r after Fisher-z transform."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    if not r.size:
        return np.nan, np.nan
    return float(np.mean(r)), float(invf(np.mean(fisher(r))))

def corr_raw_stats(r):
    """One-sample t-test on r vs 0 in raw space; returns mean r, t, p, df."""
    r = np.asarray(r, float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan
    mean_r = float(np.mean(r))
    se = np.std(r, ddof=1) / np.sqrt(n)
    tstat = 0.0 if se == 0 else mean_r / se
    df = n - 1
    p = 2 * tdist.sf(abs(tstat), df=df)
    return mean_r, float(tstat), float(p), float(df)

def z_one_sample(r):
    """One-sample test vs 0 in Fisher-z space."""
    z = fisher(r)
    z = z[np.isfinite(z)]
    n = z.size
    if n < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    m, se = np.mean(z), np.std(z, ddof=1) / np.sqrt(n)
    zstat = 0.0 if se == 0 else m / se
    p = 2 * norm.sf(abs(zstat))
    r_raw, r_z = mean_rs(r)
    return float(zstat), float(n - 1), float(p), r_raw, r_z

pooled_df = lambda x, y: float(len(x) + len(y) - 2)

# ============================================================
# 2.1 Recall ISC across conditions (event-by-event variability)
# ============================================================

# ---------- 2.1.1 Within-condition tests vs 0 (shared sections) ----------
col = "recall_pw-isc"
D = {k: pull(k, col) for k in ("free", "yoke", "pasv")}

print("2.1.1 Individual variability in recalled events. Recall ISC was significantly above zero in all three conditions.\n")

for k in ("free", "yoke", "pasv"):
    r_mean, t_raw, p_raw, df_t = corr_raw_stats(D[k])
    z, df_z, p_z, r_raw, r_z = z_one_sample(D[k])
    print(f"Romance: {LABEL[k]:7s}")
    print(f"  Raw stats:         r={r_mean:.3f}, t({int(df_t)})={t_raw:.3f}, p={p_raw:.3f}")
    print(f"  Fisher Transform:  r={r_z:.3f}, z={z:.3f}, p={p_z:.3f}")
print()

# ---------- 2.1.2 Agency effect — ANOVA + post-hoc ----------
print("2.1.2 Agency induced greater individual variability in which events were recalled.\n")

resA = anova_oneway(list(D.values()), use_var="equal")
F_eq, df1_eq, df2_eq, p_eq = resA.statistic, resA.df_num, resA.df_denom, resA.pvalue
print(f"ANOVA: F({int(df1_eq)},{int(df2_eq)}) = {F_eq:.2f}, p = {p_eq:.3f}")

pairs = [("free", "pasv"), ("free", "yoke"), ("yoke", "pasv")]
pVals = []
for a, b in pairs:
    A, B = D[a], D[b]
    tP, pP = ttest_ind(A, B, equal_var=True)
    dfP = pooled_df(A, B)
    pVals.append(pP)
    print(f"  {LABEL[a]} vs {LABEL[b]}: t({int(dfP)})={tP:.2f}, p={pP:.3f}")
pB = multipletests(pVals, method="bonferroni")[1]
print("  Bonferroni-corrected pairwise tests:")
print("    " + ", ".join(
    f"{LABEL[a]} vs {LABEL[b]}: p_bonf={pb:.3f}"
    for (a, b), pb in zip(pairs, pB)
))
print()

# ---------- 2.1.3 Same analyses excluding choice events ----------
col = "recall_pw-isc_withoutChoiceEvents"
D = {k: pull(k, col) for k in ("free", "yoke", "pasv")}

print("2.1.3 Excluding choice events, the pattern remained: Recall ISC was above zero in all conditions.\n")

for k in ("free", "yoke", "pasv"):
    r_mean, t_raw, p_raw, df_t = corr_raw_stats(D[k])
    z, df_z, p_z, r_raw, r_z = z_one_sample(D[k])
    print(f"Romance: {LABEL[k]:7s}")
    print(f"  Raw stats:         r={r_mean:.3f}, t({int(df_t)})={t_raw:.3f}, p={p_raw:.3f}")
    print(f"  Fisher Transform:  r={r_z:.3f}, z={z:.3f}, p={p_z:.3f}")
print()

resA = anova_oneway(list(D.values()), use_var="equal")
F_eq, df1_eq, df2_eq, p_eq = resA.statistic, resA.df_num, resA.df_denom, resA.pvalue
print(f"ANOVA excluding choice events: F({int(df1_eq)},{int(df2_eq)}) = {F_eq:.2f}, p = {p_eq:.3f}")

pVals = []
for a, b in pairs:
    A, B = D[a], D[b]
    tP, pP = ttest_ind(A, B, equal_var=True)
    dfP = pooled_df(A, B)
    pVals.append(pP)
    print(f"  {LABEL[a]} vs {LABEL[b]}: t({int(dfP)})={tP:.2f}, p={pP:.3f}")
pB = multipletests(pVals, method="bonferroni")[1]
print("  Bonferroni-corrected pairwise tests (excluding choice events): " +
      ", ".join(
          f"{LABEL[a]} vs {LABEL[b]}: p_bonf={pb:.3f}"
          for (a, b), pb in zip(pairs, pB)
      ))
print()

# ============================================================
# 2.2 Choice ISC (event-by-event similarity of chosen options)
# ============================================================

free = pull("free", "choice_pw-isc")
yoke = pull("yoke", "choice_pw-isc")

print("2.2.1 Choice ISC was significantly above zero in both Free and Yoked conditions.\n")

rF_mean, tF_raw, pF_raw, dfF_t = corr_raw_stats(free)
zF, dfF_z, pF_z, rF_raw2, rF_z = z_one_sample(free)
rY_mean, tY_raw, pY_raw, dfY_t = corr_raw_stats(yoke)
zY, dfY_z, pY_z, rY_raw2, rY_z = z_one_sample(yoke)

print("Romance: Free  (Choice ISC)")
print(f"  Raw stats:         r={rF_mean:.3f}, t({int(dfF_t)})={tF_raw:.3f}, p={pF_raw:.3f}")
print(f"  Fisher Transform:  r={rF_z:.3f}, z={zF:.3f}, p={pF_z:.3f}")
print("Romance: Yoked (Choice ISC)")
print(f"  Raw stats:         r={rY_mean:.3f}, t({int(dfY_t)})={tY_raw:.3f}, p={pY_raw:.3f}")
print(f"  Fisher Transform:  r={rY_z:.3f}, z={zY:.3f}, p={pY_z:.3f}")
print()

print("2.2.2 Agency (full vs partial) induced greater individual variability in which options were selected.\n")
tP, pP = ttest_ind(free, yoke, equal_var=True)
dfP = pooled_df(free, yoke)
print(f"Two-sample t-test comparing Free vs Yoked choice ISC: t({int(dfP)})={tP:.2f}, p={pP:.3f}\n")

# ============================================================
# 2.3 Correlation between memory divergence and choice divergence
# ============================================================

print("2.3 Memory divergence and choice divergence were correlated with each other.\n")

df_label = {"corr_free18": 18, "corr_free100": 100}

for sh in ("corr_free18", "corr_free100"):
    df = (pd.read_excel(XLSX, sheet_name=sh)
          [["rcl_diverg-from-group", "cho_diverg-from-group"]]
          .astype(float).dropna())
    r = float(np.corrcoef(df.iloc[:, 0], df.iloc[:, 1])[0, 1])
    n = len(df)

    if n > 2 and abs(r) < 1:
        df_corr = n - 2
        t_raw = r * np.sqrt(df_corr / (1 - r**2))
        p_raw = 2 * tdist.sf(abs(t_raw), df=df_corr)
    else:
        df_corr, t_raw, p_raw = np.nan, np.nan, np.nan

    print(f"Divergence corr ({sh}):")
    if np.isfinite(r) and np.isfinite(df_corr):
        print(f"  Raw stats: r({int(df_corr)})={r:.3f}, t({int(df_corr)})={t_raw:.3f}, p={p_raw:.3f}")
    else:
        print("  Raw stats: insufficient data")
print()
