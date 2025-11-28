"""
See manuscript results section: Agency magnified individual variability in recall and choice (choice-excluded permutation)
"""

import os, re, numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import gaussian_kde, norm
import matplotlib.pyplot as plt
from pathlib import Path

# ---------- CONFIG ----------
base_data   = Path("..") / "data"
overall_dir = base_data / "overall_data"
indiv_dir   = base_data / "individual_data"

MAP_XLSX   = overall_dir / "monthy_map.xlsx"

PASV_XLSX  = indiv_dir / "recall_prob"  / "romance" / "3_pasv" / "recall_allsub.xlsx"
PASV_EVTS  = indiv_dir / "rating" / "romance" / "3_pasv"  # {sid}_events.xlsx

YOKE_XLSX  = indiv_dir / "recall_prob" / "romance" / "2_yoke" / "recall_allsub.xlsx"
YOKE_EVTS  = indiv_dir / "rating" / "romance" / "2_yoke"  # {sid}_events.xlsx

ROM_RES_XL = overall_dir / "romance_result.xlsx"

FREE_IDS   = ["sub10","sub100","sub14","sub19","sub23","sub25","sub3","sub37","sub4",
              "sub40","sub50","sub71","sub79","sub80","sub81","sub90","sub94","sub95"]

N_PERM     = 10_000
SEED       = 13
BATCH      = 512          # batch size for vectorized draws (tune 256–2048)
# ----------------------------

# ---------- helpers ----------
def get_shared(path):
    d = pd.read_excel(path)
    return (d.assign(Converge=d["Converge"].astype(str).str.strip().str.upper(),
                     event_lab=d["event_lab"].astype(str).str.strip())
              .loc[lambda x: x["Converge"].eq("Y"), "event_lab"]
              .drop_duplicates().tolist())

def subj_pos(ev_path, shared, cand_priority=None):
    """Return {event_lab -> first idx} and the raw events df (for choice detection)."""
    if not os.path.exists(ev_path):
        raise FileNotFoundError(ev_path)
    if cand_priority is None:
        cand_priority = ["event_lab","eventlab","event label","event","event_id","eventid"]
    df = pd.read_excel(ev_path)
    df.columns = df.columns.map(lambda c: str(c).strip())
    low = {c.lower(): c for c in df.columns}
    col = next((low[c.lower()] for c in cand_priority if c.lower() in low), None)
    if col is None:
        # fallback: pick column with max overlap to shared set
        S = set(map(str, shared))
        col = max(df.columns, key=lambda c: pd.Series(df[c].astype(str).str.strip()).isin(S).sum())
    ev = df[col].astype(str).str.strip()
    first = ~ev.duplicated(keep="first")
    return {e:i for i,e in enumerate(ev[first])}, df

def find_choice_rows(events_df):
    """Indices for 'choice' rows: scenes has exactly one number and contains underscore."""
    if "scenes" not in events_df.columns:
        return set()
    s = events_df["scenes"].astype(str)
    has_one_num = s.str.count(r"\d+").fillna(0).eq(1)
    has_underscore = s.str.contains("_", regex=False)
    return set(events_df.index[has_one_num & has_underscore])

def build_matrix(recall_xlsx, events_dir, shared, patt,
                 cand_priority=None, exclude_choice=False):
    """Rows = shared events, Cols = subjects; values aligned by event_lab (0..1)."""
    R = pd.read_excel(recall_xlsx)
    R.columns = R.columns.map(lambda c: str(c).strip())
    subs = [c for c in R.columns if re.fullmatch(patt, c)]
    if not subs:
        raise ValueError(f"No subject columns match {patt!r} in {os.path.basename(recall_xlsx)}.")
    R = R[subs].apply(pd.to_numeric, errors="coerce")
    out = {}
    for sid in subs:
        pos, evdf = subj_pos(os.path.join(events_dir, f"{sid}_events.xlsx"), shared, cand_priority)
        choice_idx = find_choice_rows(evdf) if exclude_choice else set()
        vals = []
        for e in shared:
            if e in pos and pos[e] not in choice_idx:
                v = R.at[pos[e], sid]
                vals.append(float(v) if pd.notna(v) else np.nan)
            else:
                vals.append(np.nan)  # exclude missing or choice events
        out[sid] = vals
    return pd.DataFrame(out, index=shared).clip(0,1)

def bins_from_cols(cols):
    """Group IDs by Free path: 'to23_sub####' -> bin 'sub23'."""
    B = defaultdict(list)
    for c in cols:
        m = re.fullmatch(r"to(\d+)_sub\d+", c)
        if m:
            B[f"sub{m.group(1)}"].append(c)
    return dict(B)

def mean_upper_from_R(R_full, idx):
    """Mean of upper-triangle entries from a precomputed corr matrix for chosen indices."""
    sub = R_full[np.ix_(idx, idx)]
    iu = np.triu_indices(sub.shape[0], 1)
    v = sub[iu]
    v = v[~np.isnan(v)]
    return float(np.mean(v)) if v.size else np.nan

def _upper_r_values(M):
    A = M.apply(pd.to_numeric, errors="coerce").corr(method="pearson", min_periods=2).to_numpy()
    iu = np.triu_indices_from(A, 1)
    v = A[iu]
    return v[~np.isnan(v)]
# ----------------------------

# 1) shared events
shared = get_shared(MAP_XLSX)

# 2) matrices **excluding choice events**
M_pasv_nc = build_matrix(
    PASV_XLSX, PASV_EVTS, shared,
    patt=r"to\d+_sub\d+", exclude_choice=True
)
M_yoke_nc = build_matrix(
    YOKE_XLSX, YOKE_EVTS, shared,
    patt=r"to\d+_sub\d+",
    cand_priority=["old_seg","event_lab","event","event_id"],
    exclude_choice=True
)

# 3) Free mean ISC (no-choice) from results file
ISC = pd.read_excel(ROM_RES_XL, sheet_name="merge_isc_by-cond")
free_vec = (ISC.loc[ISC["cond"].astype(str).str.strip().str.lower().eq("free"),
                    "recall_pw-isc_withoutChoiceEvents"]
              .dropna().astype(float))
OBS_FREE_MEAN = float(free_vec.mean())
OBS_FREE_SEM  = float(free_vec.std(ddof=1)/np.sqrt(len(free_vec)))
print(f"Observed Free mean ISC (no-choice): {OBS_FREE_MEAN:.3f} (SEM {OBS_FREE_SEM:.3f}, n={len(free_vec)})")

# Also compute Yoked & Passive ISC from matrices (for bar plot)
vals_yoke_nc = _upper_r_values(M_yoke_nc)
vals_pasv_nc = _upper_r_values(M_pasv_nc)

OBS_YOKE_MEAN = float(np.nanmean(vals_yoke_nc)) if vals_yoke_nc.size else np.nan
OBS_YOKE_SEM  = float(np.nanstd(vals_yoke_nc, ddof=1) / np.sqrt(len(vals_yoke_nc))) if len(vals_yoke_nc) > 1 else np.nan

OBS_PASV_MEAN = float(np.nanmean(vals_pasv_nc)) if vals_pasv_nc.size else np.nan
OBS_PASV_SEM  = float(np.nanstd(vals_pasv_nc, ddof=1) / np.sqrt(len(vals_pasv_nc))) if len(vals_pasv_nc) > 1 else np.nan

print(f"Observed Yoked mean ISC (no-choice):   {OBS_YOKE_MEAN:.3f} (SEM {OBS_YOKE_SEM:.3f}, n={len(vals_yoke_nc)})")
print(f"Observed Passive mean ISC (no-choice): {OBS_PASV_MEAN:.3f} (SEM {OBS_PASV_SEM:.3f}, n={len(vals_pasv_nc)})")

# 4) permutation bins (Passive + Yoked, one-per-Free path)
FREE_IDS_low = [f.lower() for f in FREE_IDS]

# Passive bins
bins_pasv = bins_from_cols(M_pasv_nc.columns.tolist())
missing_pasv = [f for f in FREE_IDS_low if f not in bins_pasv or len(bins_pasv[f]) == 0]
if missing_pasv:
    raise ValueError(f"No Passive candidates for: {missing_pasv}")
bin_lists_pasv = [bins_pasv[f] for f in FREE_IDS_low]

# Yoked bins
bins_yoke = bins_from_cols(M_yoke_nc.columns.tolist())
missing_yoke = [f for f in FREE_IDS_low if f not in bins_yoke or len(bins_yoke[f]) == 0]
if missing_yoke:
    raise ValueError(f"No Yoked candidates for: {missing_yoke}")
bin_lists_yoke = [bins_yoke[f] for f in FREE_IDS_low]

# Precompute correlation matrices ONCE
R_pasv = M_pasv_nc.corr(method="pearson", min_periods=2).to_numpy()
pasv_idx = {sid: i for i, sid in enumerate(M_pasv_nc.columns)}

R_yoke = M_yoke_nc.corr(method="pearson", min_periods=2).to_numpy()
yoke_idx = {sid: i for i, sid in enumerate(M_yoke_nc.columns)}

# 5) permutations (10k) with batching for BOTH Passive and Yoked
rng = np.random.default_rng(SEED)
perm_pasv = np.empty(N_PERM, dtype=float)
perm_yoke = np.empty(N_PERM, dtype=float)

def draw_one_set(bin_lists):
    """Draw IDs: one per Free path bin."""
    return [rng.choice(lst) for lst in bin_lists]

filled = 0
while filled < N_PERM:
    batch = min(BATCH, N_PERM - filled)
    # Draw batch for Passive and Yoked
    draws_pasv = [draw_one_set(bin_lists_pasv) for _ in range(batch)]
    draws_yoke = [draw_one_set(bin_lists_yoke) for _ in range(batch)]

    for i, (ids_pasv, ids_yoke) in enumerate(zip(draws_pasv, draws_yoke)):
        idx_pasv = [pasv_idx[s] for s in ids_pasv]
        idx_yoke = [yoke_idx[s] for s in ids_yoke]
        perm_pasv[filled + i] = mean_upper_from_R(R_pasv, idx_pasv)
        perm_yoke[filled + i] = mean_upper_from_R(R_yoke, idx_yoke)

    filled += batch

print(f"Completed {N_PERM} permutations; Passive finite: {np.isfinite(perm_pasv).sum()}/{N_PERM}, "
      f"Yoked finite: {np.isfinite(perm_yoke).sum()}/{N_PERM}")

# Non-parametric p-values relative to Passive null
p_free_vs_pasv = (np.sum(perm_pasv <= OBS_FREE_MEAN) + 1) / (N_PERM + 1)
p_yoke_vs_pasv = (np.sum(perm_pasv <= OBS_YOKE_MEAN) + 1) / (N_PERM + 1) if np.isfinite(OBS_YOKE_MEAN) else np.nan

print(f"Passive null (no-choice): mean={perm_pasv.mean():.3f}, sd={perm_pasv.std(ddof=1):.3f}, n={N_PERM}")
print(f"Non-parametric p (Free < Passive, no-choice):  {p_free_vs_pasv:.4g}")
print(f"Non-parametric p (Yoked < Passive, no-choice): {p_yoke_vs_pasv:.4g}")

# 6) Histogram (COUNTS) with Passive + Yoked perm distributions, Free line
fig, ax = plt.subplots(figsize=(7.4, 4.6))

bins_n = 50
# Use common bins so overlap is clear
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
    label="Passive perm (no-choice)"
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
    label="Yoked perm (no-choice)"
)

# X-range for curves
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

# Free observed line (no-choice)
ax.axvline(OBS_FREE_MEAN, color="blue", linestyle="-", linewidth=2)
ax.text(
    OBS_FREE_MEAN,
    ax.get_ylim()[1] * 0.95,
    f"Free (no-choice) = {OBS_FREE_MEAN:.3f}",
    color="blue",
    ha="right",
    va="top",
    rotation=90,
    bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5)
)

ax.set_title("Passive vs Yoked permutation nulls (choice-excluded; one-per-path)\nwith Free observed ISC")
ax.set_xlabel("Mean pairwise ISC (r)")
ax.set_ylabel("Count")
ax.margins(x=0.02, y=0.05)
ax.legend()
ax.grid(alpha=0.2, linestyle=":")
plt.tight_layout()
plt.show()

# --- Bar chart: Free, Yoked, Passive (choice-excluded) ---
labels = ["Free", "Yoked", "Passive"]
means  = [OBS_FREE_MEAN, OBS_YOKE_MEAN, OBS_PASV_MEAN]
sems   = [OBS_FREE_SEM,  OBS_YOKE_SEM,  OBS_PASV_SEM]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # blue, orange, green

fig, ax = plt.subplots(figsize=(6.5, 4.2))
bars = ax.bar(labels, means, yerr=sems, capsize=6, edgecolor="black",
              linewidth=1.2, alpha=0.95, color=colors)

ax.set_ylabel("Mean pairwise ISC (r)")
ax.set_title("ISC across shared events (choice-excluded; mean ± SEM)")
top = np.nanmax([m + (s if np.isfinite(s) else 0) for m, s in zip(means, sems)])
ax.set_ylim(0, top * 1.10 if np.isfinite(top) and top > 0 else 1.0)

for b, m in zip(bars, means):
    if np.isfinite(m):
        ax.text(b.get_x() + b.get_width()/2,
                b.get_height() + (top * 0.01 if np.isfinite(top) and top > 0 else 0.01),
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)

ax.grid(axis="y", alpha=0.2, linestyle=":")
plt.tight_layout()
plt.show()
