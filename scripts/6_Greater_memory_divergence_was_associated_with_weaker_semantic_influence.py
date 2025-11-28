"""
See manuscript results section: Greater memory divergence was associated with weaker semantic influence
"""

import numpy as np, pandas as pd
from scipy.stats import pearsonr, norm
import statsmodels.api as sm
from pathlib import Path

# ---------- Load data ----------
base_data = Path("..") / "data" / "overall_data"
ROM = base_data / "romance_result.xlsx"
ADV = base_data / "adventure_result.xlsx"

# Romance: small subset (≈18) and full sample (≈100) of Free subjects
rom18    = pd.read_excel(ROM, sheet_name="corr_free18")
rom100   = pd.read_excel(ROM, sheet_name="corr_free100")

# Adventure / Romance Free-only rows for semantic vs neighbor correlations
adv_free = pd.read_excel(ADV, sheet_name="comp_conds").query("cond=='free'")
rom18_f  = pd.read_excel(ROM, sheet_name="comp_conds18").query("cond=='free'")


def _num(s):
    """Safe numeric conversion (returns NaN for non-numeric entries)."""
    return pd.to_numeric(s, errors="coerce")


def corr_report(x, y, df_label=None):
    """
    Compact 2-line correlation report:

      Raw stats:        r(df)=..., t=..., p=...
      Fisher Transform: r(df)=..., z=..., p=...

    df_label controls what appears in r(df). If None, df defaults to n-2.
    """
    xy = pd.concat([_num(x), _num(y)], axis=1).dropna()
    if xy.shape[0] < 3:
        return (
            "  Raw stats:        r(N/A)=nan, t=nan, p=nan\n"
            "  Fisher Transform: r(N/A)=nan, z=nan, p=nan"
        )

    X = xy.iloc[:, 0].values
    Y = xy.iloc[:, 1].values
    n = len(X)

    # Pearson correlation
    r, p_raw = pearsonr(X, Y)

    # t statistic for Pearson r with df = n-2
    if abs(r) < 1 and n > 2:
        t_raw = r * np.sqrt((n - 2) / (1 - r**2))
    else:
        t_raw = np.nan

    # Fisher-z transform (approximate z-test for r ≠ 0)
    zr = np.arctanh(np.clip(r, -0.999999, 0.999999))
    z  = zr * np.sqrt(max(n - 3, 1))
    p_z = 2 * norm.sf(abs(z))

    # Label in r(df): by default use df = n-2
    df_val = (n - 2) if df_label is None else df_label

    text  = f"  Raw stats:        r({df_val})={r:.3f}, t={t_raw:.3f}, p={p_raw:.3f}\n"
    text += f"  Fisher Transform: r({df_val})={r:.3f}, z={z:.3f}, p={p_z:.3f}"
    return text


def ols_mem_div(df):
    """
    Multiple regression of memory divergence on semantic and neighbor effects:
        rcl_diverg-from-group ~ sem-ef + nghb-ef
    Returns n, R², and p-values for each predictor.
    """
    Z = df.copy()
    Z["r"]   = _num(Z["rcl_diverg-from-group"])
    Z["sem"] = _num(Z["sem-ef"])
    Z["ngb"] = _num(Z["nghb-ef"])
    Z = Z[["r", "sem", "ngb"]].dropna()

    y = Z["r"].values
    X = sm.add_constant(Z[["sem", "ngb"]].values)
    m = sm.OLS(y, X).fit()

    return {
        "n":      int(m.nobs),
        "R2":     m.rsquared,
        "p_sem":  float(m.pvalues[1]),
        "p_nghb": float(m.pvalues[2]),
    }


# ---------- 6.1 Memory divergence ~ Semantic (Romance Free) ----------
print("6.1 Memory divergence was negatively correlated with semantic influence in Free participants.\n")

print("Romance (n≈18):")
print(corr_report(rom18["rcl_diverg-from-group"], rom18["sem-ef"]))
print()

print("Romance (n≈100):")
print(corr_report(rom100["rcl_diverg-from-group"], rom100["sem-ef"]))
print()


# ---------- 6.2 Memory divergence ~ Neighbor encoding (Romance Free) ----------
print("6.2 Neighbor encoding effect was positively associated with memory divergence in Free participants.\n")

print("Romance (n≈18):")
print(corr_report(rom18["rcl_diverg-from-group"], rom18["nghb-ef"]))
print()

print("Romance (n≈100):")
print(corr_report(rom100["rcl_diverg-from-group"], rom100["nghb-ef"]))
print()


# ---------- 6.3 Neighbor encoding ~ Semantic influence (Free, both stories) ----------
print("6.3 Neighbor encoding effect was negatively correlated with semantic influence in Free participants (both stories).\n")

print("Adventure (Free):")
print(corr_report(adv_free["nghb-ef"], adv_free["sem-ef"]))
print()

print("Romance (n≈18):")
print(corr_report(rom18_f["nghb-ef"], rom18_f["sem-ef"]))
print()

print("Romance (n≈100):")
print(corr_report(rom100["nghb-ef"], rom100["sem-ef"]))
print()


# ---------- 6.4 Multiple regression: mem_div ~ sem-ef + nghb-ef (Romance Free) ----------
print("6.4 When including both semantic influence and the neighbor encoding scores in a multiple linear regression predicting memory divergence, semantic influence score was a significant predictor while the neighbor encoding effect only showed a trend.\n")

o18  = ols_mem_div(rom18)
o100 = ols_mem_div(rom100)

print(f"Romance (n≈18):  R^2={o18['R2']:.3f};  sem-ef p={o18['p_sem']:.3f}; nghb-ef p={o18['p_nghb']:.3f}")
print(f"Romance (n≈100): R^2={o100['R2']:.3f}; sem-ef p={o100['p_sem']:.3f}; nghb-ef p={o100['p_nghb']:.3f}\n")
