"""
See manuscript results section: Agency reduced the influence of semantic but not causal centrality on recall
"""

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, ttest_1samp, norm
from pathlib import Path

import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

# ---------- Cross-platform paths ----------
base_data = Path("..") / "data" / "overall_data"
FILES = {
    "Adventure": base_data / "adventure_result.xlsx",
    "Romance":   base_data / "romance_result.xlsx",
}

# Conditions and labels
CONDS = ("free", "yoke", "pasv")
COND_LABEL = {"free": "Free", "yoke": "Yoked", "pasv": "Passive"}

# Group helper for one-way ANOVA
grp = lambda df, col: [
    df.loc[df["cond"].eq(c), col].dropna().astype(float).values
    for c in CONDS
]

def classic_anova(G):
    """Return df1, df2, F, p for equal-variance one-way ANOVA across 3 conditions."""
    k = 3
    n = sum(len(g) for g in G)
    F, p = f_oneway(*G)
    return (k - 1, n - k, F, p)

# ---------- Fisher-style helpers (treat values as correlation-like) ----------
fisher = lambda x: np.arctanh(np.clip(np.asarray(x, float), -0.999999, 0.999999))
invf   = lambda z: np.tanh(np.asarray(z, float))

def fisher_one_sample(x):
    """
    One-sample test in Fisher-z space (vs 0), analogous to correlation tests.
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

    m  = np.mean(z)
    se = np.std(z, ddof=1) / np.sqrt(n)
    zstat = 0.0 if se == 0 else m / se
    p = 2 * norm.sf(abs(zstat))

    mean_raw  = float(np.mean(x))
    mean_back = float(invf(m))
    return float(zstat), float(n - 1), float(p), mean_raw, mean_back


# ====================================================
# 4.1 Semantic influence (between-subjects ANOVA)
# ====================================================
print("4.1 Observed significant differences across conditions, wherein Free had lower semantic influence on memory compared to Yoked and Passive.\n")

for story, fp in FILES.items():
    df = pd.read_excel(fp)

    print(f"{story} — Semantic")
    # Per-condition raw & Fisher-z stats
    for c in CONDS:
        s = df.loc[df["cond"].eq(c), "sem-ef"].dropna().astype(float).values
        n = len(s)
        if n < 2:
            continue

        # Raw one-sample t-test vs 0
        t_raw, p_raw = ttest_1samp(s, 0)
        mean_raw = float(np.mean(s))

        # Fisher-z based one-sample test vs 0
        zstat, df_z, p_z, mean_raw_f, mean_back = fisher_one_sample(s)

        print(f"  {COND_LABEL[c]}:")
        print(f"    Raw stats:        mean={mean_raw:.3f}, t({n-1})={t_raw:.3f}, p={p_raw:.3f}")
        print(f"    Fisher Transform: mean_z={mean_back:.3f}, z({int(df_z)})={zstat:.3f}, p={p_z:.3f}")

    # Between-subjects ANOVA across conditions (one-way)
    G = grp(df, "sem-ef")
    d1, d2, F, p = classic_anova(G)
    print(f"  ANOVA (Semantic):   F({int(d1)},{int(d2)}) = {F:.2f}, p = {p:.3f}\n")

print()


# ====================================================
# 4.2 Causal influence (between-subjects ANOVA)
# ====================================================
print("4.2 Causal influence on memory did not differ across agency conditions.\n")

for story, fp in FILES.items():
    df = pd.read_excel(fp)

    print(f"{story} — Causal")
    # Per-condition raw & Fisher-z stats
    for c in CONDS:
        s = df.loc[df["cond"].eq(c), "caus-ef"].dropna().astype(float).values
        n = len(s)
        if n < 2:
            continue

        # Raw one-sample t-test vs 0
        t_raw, p_raw = ttest_1samp(s, 0)
        mean_raw = float(np.mean(s))

        # Fisher-z based one-sample test vs 0
        zstat, df_z, p_z, mean_raw_f, mean_back = fisher_one_sample(s)

        print(f"  {COND_LABEL[c]}:")
        print(f"    Raw stats:        mean={mean_raw:.3f}, t({n-1})={t_raw:.3f}, p={p_raw:.3f}")
        print(f"    Fisher Transform: mean_z={mean_back:.3f}, z({int(df_z)})={zstat:.3f}, p={p_z:.3f}")

    # Between-subjects ANOVA across conditions (one-way)
    G = grp(df, "caus-ef")
    d1, d2, F, p = classic_anova(G)
    print(f"  ANOVA (Causal):     F({int(d1)},{int(d2)}) = {F:.2f}, p = {p:.3f}\n")

print()


# ====================================================
# 4.3 Two-way ANOVA: Network Type (Semantic vs Causal) × Condition
# ====================================================
print("4.3 Two-way ANOVA (Network Type × Condition) on centrality effects.\n")
print("There was a significant network type × agency interaction for the Adventure story and a trend for the same for the Romance story.\n")

for story, fp in FILES.items():
    df = pd.read_excel(fp)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Need subject ID, condition, and both centrality effects
    if not {"subj", "cond", "sem-ef", "caus-ef"}.issubset(df.columns):
        continue

    # Long-format: one row per (subject, condition, network type)
    sem_rows = df[["subj", "cond", "sem-ef"]].copy()
    sem_rows = sem_rows.rename(columns={"sem-ef": "effect"})
    sem_rows["network"] = "semantic"

    caus_rows = df[["subj", "cond", "caus-ef"]].copy()
    caus_rows = caus_rows.rename(columns={"caus-ef": "effect"})
    caus_rows["network"] = "causal"

    long_df = pd.concat([sem_rows, caus_rows], ignore_index=True)
    long_df = long_df.dropna(subset=["effect"])

    long_df["cond"]    = long_df["cond"].astype(str).str.strip().str.lower()
    long_df["network"] = long_df["network"].astype(str)
    long_df["subj"]    = long_df["subj"].astype(str)

    if long_df.empty:
        continue

    # OLS with subject as a fixed effect: effect ~ Condition * Network + Subject
    model = smf.ols("effect ~ C(cond) * C(network) + C(subj)", data=long_df).fit()
    aov   = anova_lm(model, typ=2)

    print(f"{story} — Two-way ANOVA (Condition × Network):")
    for term, label in [
        ("C(cond)", "Condition"),
        ("C(network)", "Network"),
        ("C(cond):C(network)", "Condition × Network"),
    ]:
        if term in aov.index:
            df1   = aov.loc[term, "df"]
            F_val = aov.loc[term, "F"]
            p_val = aov.loc[term, "PR(>F)"]
            if "Residual" in aov.index:
                df2 = aov.loc["Residual", "df"]
            else:
                df2 = model.df_resid
            print(f"  {label}: F({int(df1)},{int(df2)}) = {F_val:.2f}, p = {p_val:.3f}")
    print()
