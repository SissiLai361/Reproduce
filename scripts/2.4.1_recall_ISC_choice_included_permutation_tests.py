"""
See manuscript results section: Agency magnified individual variability in recall and choice
"""

import os, re, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import gaussian_kde, norm
from pathlib import Path

# ---------- CONFIG ----------
base_data   = Path("..") / "data"
overall_dir = base_data / "overall_data"
indiv_dir   = base_data / "individual_data"

MAP_XLSX   = overall_dir / "monthy_map.xlsx"
PASV_XLSX  = indiv_dir / "recall_prob" / "romance" / "3_pasv" / "recall_allsub.xlsx"
PASV_EVTS  = indiv_dir / "rating" / "romance"/ "3_pasv"    # {sid}_events.xlsx (to23_sub6001)
YOKE_XLSX  = indiv_dir / "recall_prob" / "romance" / "2_yoke" / "recall_allsub.xlsx"
YOKE_EVTS  = indiv_dir / "rating" / "romance" / "2_yoke"   # {sid}_events.xlsx (to23_sub5001), uses old_seg
ROM_RES_XL = overall_dir / "romance_result.xlsx"           # for Free ISC directly

FREE_IDS   = ["sub10","sub100","sub14","sub19","sub23","sub25","sub3","sub37","sub4",
              "sub40","sub50","sub71","sub79","sub80","sub81","sub90","sub94","sub95"]
N_PERM, SEED = 10_000, 13
# ----------------------------

# ---- helpers (tight & robust) ----
def get_shared(path):
    d = pd.read_excel(path)
    d["Converge"] = d["Converge"].astype(str).str.strip().str.upper()
    d["event_lab"] = d["event_lab"].astype(str).str.strip()
    s = d.loc[d["Converge"].eq("Y"), "event_lab"].drop_duplicates().tolist()
    if not s:
        raise ValueError("No shared events (Converge=='Y').")
    return s

def subj_pos(ev_path, shared, cand_priority=None):
    if not os.path.exists(ev_path):
        raise FileNotFoundError(ev_path)
    if cand_priority is None:
        cand_priority = ["event_lab","eventlab","event label","event","event_id","eventid"]
    df = pd.read_excel(ev_path, header=0)
    df.columns = df.columns.map(lambda x: str(x).strip())
    low = {c.lower(): c for c in df.columns}
    cand = next((low[c.lower()] for c in cand_priority if c.lower() in low), None)
    if cand is None and len(df):
        # header-in-first-row rescue
        hdr = df.iloc[0].astype(str).str.strip().str.lower().fillna("")
        if any(h in [c.lower() for c in cand_priority] for h in hdr):
            df.columns = df.iloc[0].astype(str).str.strip()
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = df.columns.map(lambda x: str(x).strip())
            low = {c.lower(): c for c in df.columns}
            cand = next((low[c.lower()] for c in cand_priority if c.lower() in low), None)
    if cand is None:
        # fallback: column with max overlap with shared labels
        S = set(map(str, shared))
        cand = max(
            df.columns,
            key=lambda c: pd.Series(df[c].astype(str).str.strip()).isin(S).sum()
        )
    ev = df[cand].astype(str).str.strip()
    first = ~ev.duplicated(keep="first")
    return {e: i for i, e in enumerate(ev[first])}

def build_matrix(recall_xlsx, events_dir, shared, patt, cand_priority=None):
    R = pd.read_excel(recall_xlsx, header=0)
    R.columns = R.columns.map(lambda x: str(x).strip())
    subs = [c for c in R.columns if re.fullmatch(patt, c)]
    if not subs and len(R):
        # header-in-first-row rescue
        hdr = R.iloc[0].astype(str).str.strip().fillna("")
        if any(re.fullmatch(patt, h) for h in hdr):
            R.columns = hdr
            R = R.iloc[1:].reset_index(drop=True)
            R.columns = R.columns.map(lambda x: str(x).strip())
            subs = [c for c in R.columns if re.fullmatch(patt, c)]
    if not subs:
        raise ValueError(f"No subject columns match {patt!r} in {os.path.basename(recall_xlsx)}.")
    R = R[subs].apply(pd.to_numeric, errors="coerce")
    out = {}
    for sid in subs:
        ev_file = os.path.join(events_dir, f"{sid}_events.xlsx")
        pos = subj_pos(ev_file, shared, cand_priority)
        out[sid] = [
            float(R.at[pos[e], sid]) if (e in pos and pd.notna(R.at[pos[e], sid])) else np.nan
            for e in shared
        ]
    return pd.DataFrame(out, index=shared).clip(0, 1)

def mean_upper_r(M):
    A = M.apply(pd.to_numeric, errors="coerce").corr(method="pearson", min_periods=2).to_numpy()
    iu = np.triu_indices_from(A, 1)
    v  = A[iu]
    v  = v[~np.isnan(v)]
    return float(np.mean(v)) if len(v) else np.nan

def upper_r_values(M):
    A = M.apply(pd.to_numeric, errors="coerce").corr(method="pearson", min_periods=2).to_numpy()
    iu = np.triu_indices_from(A, 1)
    v  = A[iu]
    return v[~np.isnan(v)]

def bins_from_cols(cols):
    B = defaultdict(list)
    for c in cols:
        m = re.fullmatch(r"to(\d+)_sub\d+", c)
        if m:
            B[f"sub{m.group(1)}"].append(c)
    return dict(B)
# ----------------------------

# 1) Shared events (ordered)
shared = get_shared(MAP_XLSX)

# 2) Build matrices for Passive and Yoked (Yoked prefers 'old_seg')
M_pasv = build_matrix(PASV_XLSX, PASV_EVTS, shared, patt=r"to\d+_sub\d+")
M_yoke = build_matrix(
    YOKE_XLSX,
    YOKE_EVTS,
    shared,
    patt=r"to\d+_sub\d+",
    cand_priority=["old_seg","event_lab","event","event_id"]
)

# 3) Observed Free ISC directly from romance_result.xlsx
ISC = pd.read_excel(ROM_RES_XL, sheet_name="merge_isc_by-cond")
free_vec = ISC.loc[
    ISC["cond"].astype(str).str.strip().str.lower().eq("free"),
    "recall_pw-isc"
].dropna().astype(float)
OBS_FREE_MEAN = float(free_vec.mean())
OBS_FREE_SEM  = float(free_vec.std(ddof=1) / np.sqrt(len(free_vec)))
print(f"Observed Free mean ISC (from romance_result.xlsx): {OBS_FREE_MEAN:.3f} "
      f"(SEM {OBS_FREE_SEM:.3f}, n={len(free_vec)})")

# Also compute Yoked & Passive ISC from matrices (for bar plot)
vals_yoke = upper_r_values(M_yoke)
vals_pasv = upper_r_values(M_pasv)

OBS_YOKE_MEAN = float(np.mean(vals_yoke)) if len(vals_yoke) else np.nan
OBS_YOKE_SEM  = float(np.std(vals_yoke, ddof=1) / np.sqrt(len(vals_yoke))) if len(vals_yoke) else np.nan

OBS_PASV_MEAN = float(np.mean(vals_pasv)) if len(vals_pasv) else np.nan
OBS_PASV_SEM  = float(np.std(vals_pasv, ddof=1) / np.sqrt(len(vals_pasv))) if len(vals_pasv) else np.nan

print(f"Observed Yoked mean ISC (from M_yoke): {OBS_YOKE_MEAN:.3f} "
      f"(SEM {OBS_YOKE_SEM:.3f}, n={len(vals_yoke)})")
print(f"Observed Passive mean ISC (from M_pasv): {OBS_PASV_MEAN:.3f} "
      f"(SEM {OBS_PASV_SEM:.3f}, n={len(vals_pasv)})")

# 4) Permutation bins (Passive + Yoked): one candidate per Free path
FREE_IDS = [f.lower() for f in FREE_IDS]

# Passive bins
bins_pasv = bins_from_cols(M_pasv.columns.tolist())
missing_pasv = [f for f in FREE_IDS if f not in bins_pasv or len(bins_pasv[f]) == 0]
if missing_pasv:
    raise ValueError(f"No Passive candidates for: {missing_pasv}")
bin_lists_pasv = [bins_pasv[f] for f in FREE_IDS]

# Yoked bins
bins_yoke = bins_from_cols(M_yoke.columns.tolist())
missing_yoke = [f for f in FREE_IDS if f not in bins_yoke or len(bins_yoke[f]) == 0]
if missing_yoke:
    raise ValueError(f"No Yoked candidates for: {missing_yoke}")
bin_lists_yoke = [bins_yoke[f] for f in FREE_IDS]

# 5) Permutations (EXACTLY 10k) for both Passive and Yoked
rng = np.random.default_rng(SEED)
perm_pasv = np.empty(N_PERM, float)
perm_yoke = np.empty(N_PERM, float)

for k in range(N_PERM):
    # Passive
    chosen_pasv = [rng.choice(lst) for lst in bin_lists_pasv]
    perm_pasv[k] = mean_upper_r(M_pasv[chosen_pasv])

    # Yoked
    chosen_yoke = [rng.choice(lst) for lst in bin_lists_yoke]
    perm_yoke[k] = mean_upper_r(M_yoke[chosen_yoke])

assert perm_pasv.size == N_PERM and perm_yoke.size == N_PERM
print(f"Completed {N_PERM} permutations; Passive finite: {np.isfinite(perm_pasv).sum()}/{N_PERM}, "
      f"Yoked finite: {np.isfinite(perm_yoke).sum()}/{N_PERM}")

# Non-parametric p-values relative to Passive null
p_free_vs_pasv = (np.sum(perm_pasv <= OBS_FREE_MEAN) + 1) / (N_PERM + 1)
p_yoke_vs_pasv = (np.sum(perm_pasv <= OBS_YOKE_MEAN) + 1) / (N_PERM + 1) if np.isfinite(OBS_YOKE_MEAN) else np.nan

print(f"Passive null (perm): mean={perm_pasv.mean():.3f}, sd={perm_pasv.std(ddof=1):.3f}, n={N_PERM}")
print(f"Non-parametric p (Free < Passive):  {p_free_vs_pasv:.4g}")
print(f"Non-parametric p (Yoked < Passive): {p_yoke_vs_pasv:.4g}")

# 6) Histogram (COUNTS on y-axis) with Passive + Yoked distributions, Free line
fig, ax = plt.subplots(figsize=(7.4, 4.6))

bins_n = 50
# Common bins so overlap is comparable
all_vals = np.concatenate([perm_pasv, perm_yoke])
bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins_n + 1)
bin_width = bin_edges[1] - bin_edges[0]
N = len(perm_pasv)

# Passive histogram (green)
ax.hist(
    perm_pasv,
    bins=bin_edges,
    density=False,
    alpha=0.7,
    color="green",
    edgecolor="black",
    linewidth=0.8,
    label="Passive perm"
)

# Yoked histogram (orange)
ax.hist(
    perm_yoke,
    bins=bin_edges,
    density=False,
    alpha=0.5,
    color="orange",
    edgecolor="black",
    linewidth=0.8,
    label="Yoked perm"
)

# x-range for curves
x_min = min(all_vals.min(), OBS_FREE_MEAN)
x_max = max(all_vals.max(), OBS_FREE_MEAN)
x = np.linspace(x_min, x_max, 600)

# KDE for Passive (scaled to counts)
kde_pasv = gaussian_kde(perm_pasv)
ax.plot(x, kde_pasv(x) * N * bin_width, linewidth=2, color="darkgreen",
        label="KDE (Passive perm)")

# KDE for Yoked (scaled to counts)
kde_yoke = gaussian_kde(perm_yoke)
ax.plot(x, kde_yoke(x) * N * bin_width, linewidth=2, color="darkorange",
        label="KDE (Yoked perm)")

# Optional Normal approx for Passive
mu_pasv, sd_pasv = perm_pasv.mean(), perm_pasv.std(ddof=1)
ax.plot(x, norm.pdf(x, mu_pasv, sd_pasv) * N * bin_width,
        ":", linewidth=2, color="black", label="Normal approx (Passive)")

# Reference line for Free only
ax.axvline(OBS_FREE_MEAN, color="blue", linestyle="-", linewidth=2)
ax.text(
    OBS_FREE_MEAN,
    ax.get_ylim()[1] * 0.95,
    f"Free = {OBS_FREE_MEAN:.3f}",
    color="blue",
    ha="right",
    va="top",
    rotation=90,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5)
)

ax.set_title("Permutation nulls (one-per-path; shared events)\nPassive vs Yoked, with Free observed mean")
ax.set_xlabel("Mean pairwise ISC (r)")
ax.set_ylabel("Count")
ax.margins(x=0.02, y=0.05)
ax.legend()
ax.grid(alpha=0.2, linestyle=":")
plt.tight_layout()
plt.show()
