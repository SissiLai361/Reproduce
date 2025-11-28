"""
See manuscript results section: Choice ISC did not drive Recall ISC.

Pipeline:
- Build Free/Yoked/Passive recall matrices on shared events.
- Build Free/Yoked choice matrices on shared-choice events.
- Compute Free baselines (Choice & Recall ISC) in Fisher-z space.
- Permute Yoked: sample one Yoked subject per Free path (18 total).
- Accept a sample if mean_z(Yoked Choice) <= mean_z(Free Choice).
- For accepted samples, record mean Yoked Recall ISC.
- Build null distribution and compare Free Recall ISC to this null.
- Plot permutation null and Recall ISC bar chart (Free/Yoked/Passive).
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from math import sqrt
from pathlib import Path
from scipy.stats import t as tdist
from scipy.stats import ttest_ind, gaussian_kde, norm

# ======================= CONFIG (Romance story) =======================
base_overall = Path("..") / "data" / "overall_data"
base_indiv   = Path("..") / "data" / "individual_data"

MAP_XLSX = base_overall / "monthy_map.xlsx"

# Romance-specific recall and rating folders
recall_base = base_indiv / "recall_prob" / "romance"
rating_base = base_indiv / "rating"      / "romance"

RECALL_FREE_XLSX = recall_base / "1_free" / "recall_allsub.xlsx"
RECALL_YOKE_XLSX = recall_base / "2_yoke" / "recall_allsub.xlsx"
RECALL_PASV_XLSX = recall_base / "3_pasv" / "recall_allsub.xlsx"

EVENTS_FREE_DIR  = rating_base / "1_free"  # sub10_4001_events.xlsx
EVENTS_YOKE_DIR  = rating_base / "2_yoke"  # to23_sub5001_events.xlsx
EVENTS_PASV_DIR  = rating_base / "3_pasv"  # to23_sub6001_events.xlsx

FREE_PATHS = [
    "sub10","sub100","sub14","sub19","sub23","sub25","sub3","sub37","sub4",
    "sub40","sub50","sub71","sub79","sub80","sub81","sub90","sub94","sub95"
]

N_ACCEPT  = 10_000
SEED      = 13
MAX_TRIES = 400_000
EPS_Z     = 1e-12

# ======================= HELPERS =======================
def fisher(r):
    r = np.asarray(r, float)
    r = np.clip(r, -0.999999, 0.999999)
    return np.arctanh(r)

def inv_fisher(z):
    return np.tanh(z)

def delta_sem_r(mean_r, sem_z):
    """Delta-method SEM in r-space from SEM in z-space."""
    if not (np.isfinite(mean_r) and np.isfinite(sem_z)):
        return np.nan
    return (1 - mean_r**2) * sem_z

def _strip_cols(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df

def get_shared_recall_events(path: Path):
    """Shared recall events: Converge=='Y'."""
    m = pd.read_excel(path)
    m["Converge"]  = m["Converge"].astype(str).str.strip().str.upper()
    m["event_lab"] = m["event_lab"].astype(str).str.strip()
    return m.loc[m["Converge"].eq("Y"), "event_lab"].drop_duplicates().tolist()

def get_shared_choice_events(path: Path):
    """Shared choice events: Converge=='Y' & Scene_lab contains '_'."""
    m = pd.read_excel(path)
    for c in ("Converge","Scene_lab","event_lab"):
        if c not in m.columns:
            raise ValueError(f"{c} missing in {path}")
    mm = m.copy()
    mm["Converge"]  = mm["Converge"].astype(str).str.strip().str.upper()
    mm["Scene_lab"] = mm["Scene_lab"].astype(str).str.strip()
    mm["event_lab"] = mm["event_lab"].astype(str).str.strip()
    mask = mm["Converge"].eq("Y") & mm["Scene_lab"].str.contains("_", regex=False)
    return mm.loc[mask, "event_lab"].drop_duplicates().tolist()

def subj_pos(ev_path: Path, shared, prefer_cols=("event_lab","eventlab","event","event_id")):
    """Map event_lab -> first row index for a subject's *_events.xlsx."""
    if not ev_path.exists():
        raise FileNotFoundError(ev_path)
    df = _strip_cols(pd.read_excel(ev_path))
    if "event_lab" not in df.columns:
        hdr = [str(x).strip() for x in df.iloc[0].tolist()]
        if any(h.lower() in [c.lower() for c in prefer_cols] for h in hdr):
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = [str(x).strip() for x in hdr]
            df = _strip_cols(df)
    low = {c.lower(): c for c in df.columns}
    col = next((low[c.lower()] for c in prefer_cols if c.lower() in low), None)
    if col is None:
        S = set(map(str, shared))
        col = max(df.columns, key=lambda c: pd.Series(df[c].astype(str).str.strip()).isin(S).sum())
    ev = df[col].astype(str).str.strip()
    first = ~ev.duplicated(keep="first")
    return {e: i for e, i in zip(ev[first], np.where(first)[0])}

def build_recall_matrix(recall_xlsx: Path, events_dir: Path, shared, patt: str):
    """Shared-event recall matrix: rows=events, cols=subjects."""
    R = pd.read_excel(recall_xlsx)
    R.columns = R.columns.map(str)
    subs = [c for c in R.columns if re.fullmatch(patt, c)]
    if not subs and len(R):
        hdr = R.iloc[0].astype(str).str.strip()
        if any(re.fullmatch(patt, h) for h in hdr):
            R = R.iloc[1:].reset_index(drop=True)
            R.columns = hdr
            subs = [c for c in R.columns if re.fullmatch(patt, c)]
    if not subs:
        raise ValueError(f"No subject columns match {patt!r} in {recall_xlsx.name}")
    R = R[subs].apply(pd.to_numeric, errors="coerce")
    out = {}
    for sid in subs:
        ev_path = events_dir / f"{sid}_events.xlsx"
        pos = subj_pos(ev_path, shared)
        out[sid] = [
            float(R.at[pos[e], sid]) if (e in pos and pd.notna(R.at[pos[e], sid])) else np.nan
            for e in shared
        ]
    return pd.DataFrame(out, index=shared).clip(0, 1)

_SCENE_RE = re.compile(r"\b\d+_(\d+)\b")

def _choice_from_scenes(series: pd.Series) -> pd.Series:
    """Derive choices (1/2) from scene labels like '4_2' when only one token."""
    s = series.astype(str).str.strip()
    one = ~s.str.contains(r"\s") & s.str.contains(r"\d+_\d+")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out.loc[one] = s.loc[one].str.extract(_SCENE_RE)[0].astype(float)
    return out.where(out.isin([1.0, 2.0]))

def read_events_choice_any(ev_path: Path):
    """Return (event_lab, choice) for a subject; prefer subj_choice, else from scenes."""
    df = _strip_cols(pd.read_excel(ev_path))
    if "event_lab" not in df.columns and len(df):
        hdr = [str(x).strip() for x in df.iloc[0].tolist()]
        if any(h.lower() in ("event_lab","eventlab","event") for h in hdr):
            df = df.iloc[1:].reset_index(drop=True)
            df.columns = [str(x).strip() for x in hdr]
            df = _strip_cols(df)
    low = {c.lower(): c for c in df.columns}
    if "subj_choice" in low:
        s = pd.to_numeric(df[low["subj_choice"]], errors="coerce").where(lambda x: x.isin([1, 2]))
    else:
        sc = low.get("scenes")
        if sc is None:
            raise ValueError(f"No 'subj_choice' or 'scenes' in {ev_path}")
        s = _choice_from_scenes(df[sc])
    evc = low.get("event_lab") or low.get("eventlab") or low.get("event") or list(df.columns)[0]
    ev  = df[evc].astype(str).str.strip()
    return ev, s

def build_choice_matrix(events_dir: Path, ids, shared_choice):
    """Shared-choice matrix: rows=events, cols=subjects, values in {1,2}."""
    cols = {}
    for sid in ids:
        ev_path = events_dir / f"{sid}_events.xlsx"
        ev, choice = read_events_choice_any(ev_path)
        pos = {e: i for i, e in enumerate(ev)}
        cols[sid] = [
            float(choice.iat[pos[e]]) if (e in pos and pd.notna(choice.iat[pos[e]])) else np.nan
            for e in shared_choice
        ]
    return pd.DataFrame(cols, index=shared_choice)

def upper_r_values(M: pd.DataFrame) -> np.ndarray:
    """Upper-triangle Pearson r values (pairwise deletion) across columns."""
    A = M.apply(pd.to_numeric, errors="coerce").corr(method="pearson", min_periods=2).to_numpy()
    iu = np.triu_indices_from(A, 1)
    v  = A[iu]
    return v[~np.isnan(v)]

def mean_upper_r(M: pd.DataFrame) -> float:
    v = upper_r_values(M)
    return float(np.mean(v)) if v.size else np.nan

def bins_from_yoke_cols(cols):
    """Group Yoked IDs by Free path: 'to23_sub5001' -> bin 'sub23'."""
    B = defaultdict(list)
    for c in cols:
        m = re.fullmatch(r"to(\d+)_sub5\d+", c)
        if m:
            B[f"sub{m.group(1)}"].append(c)
    return dict(B)

def uppvec_from_full(R_full, chosen_ids, idx_lookup):
    """Upper-triangle r-vector from a full corr matrix for a chosen subject set."""
    idx = [idx_lookup[s] for s in chosen_ids]
    sub = R_full[np.ix_(idx, idx)]
    iu = np.triu_indices(len(idx), 1)
    v = sub[iu]
    return v[~np.isnan(v)]

def welch_p_z(m1, v1, n1, m2, v2, n2):
    """Two-sided Welch t-test p-value in z-space (means/vars/counts)."""
    if n1 < 2 or n2 < 2:
        return 1.0
    denom = v1/n1 + v2/n2
    if denom <= 0:
        return 1.0
    tval = (m1 - m2) / sqrt(denom)
    df = (denom**2) / ((v1*v1)/(n1*n1*(n1-1) + 1e-12) + (v2*v2)/(n2*n2*(n2-1) + 1e-12))
    df = max(1.0, float(df))
    return 2 * tdist.sf(abs(tval), df)

# ======================= LOAD & ALIGN ======================
shared_recall = get_shared_recall_events(MAP_XLSX)
shared_choice = get_shared_choice_events(MAP_XLSX)

M_free_recall = build_recall_matrix(RECALL_FREE_XLSX,  EVENTS_FREE_DIR, shared_recall, patt=r"sub\d+_4\d+")
M_yoke_recall = build_recall_matrix(RECALL_YOKE_XLSX,  EVENTS_YOKE_DIR, shared_recall, patt=r"to\d+_sub5\d+")
M_pasv_recall = build_recall_matrix(RECALL_PASV_XLSX,  EVENTS_PASV_DIR, shared_recall, patt=r"to\d+_sub6\d+")

free_ids = M_free_recall.columns.tolist()
yoke_ids = M_yoke_recall.columns.tolist()

M_free_choice = build_choice_matrix(EVENTS_FREE_DIR, free_ids, shared_choice)
M_yoke_choice = build_choice_matrix(EVENTS_YOKE_DIR, yoke_ids, shared_choice)

# ======================= FREE BASELINES =====================
FREE_CHOICE_RS = upper_r_values(M_free_choice)
FREE_CHOICE_ZS = fisher(FREE_CHOICE_RS)
FREE_CHOICE_MEAN_Z = float(np.mean(FREE_CHOICE_ZS))
FREE_CHOICE_MEAN_R = float(np.tanh(FREE_CHOICE_MEAN_Z))

FREE_RECALL_RS = upper_r_values(M_free_recall)
FREE_RECALL_MEAN = float(np.mean(FREE_RECALL_RS))
FREE_RECALL_ZS = fisher(FREE_RECALL_RS)
FREE_RECALL_SEM_Z = float(np.std(FREE_RECALL_ZS, ddof=1) / np.sqrt(max(1, len(FREE_RECALL_ZS))))
FREE_RECALL_SEM_R = delta_sem_r(FREE_RECALL_MEAN, FREE_RECALL_SEM_Z)

print(f"Free Choice ISC: mean_r={FREE_CHOICE_MEAN_R:.3f}, mean_z={FREE_CHOICE_MEAN_Z:.3f}, n_pairs={len(FREE_CHOICE_RS)}")
print(f"Free Recall ISC: mean_r={FREE_RECALL_MEAN:.3f} (SEM_z={FREE_RECALL_SEM_Z:.3f}, SEM_r≈{FREE_RECALL_SEM_R:.3f})")

bins = bins_from_yoke_cols(yoke_ids)
missing = [p for p in FREE_PATHS if p not in bins or len(bins[p]) == 0]
if missing:
    raise ValueError(f"Missing Yoked candidates for paths: {missing}")
bin_lists = [bins[p] for p in FREE_PATHS]

# ======================= PERMUTATION =======================
R_choice_yoke = M_yoke_choice.corr(min_periods=2).to_numpy()
R_recall_yoke = M_yoke_recall.corr(min_periods=2).to_numpy()

yoke_idx_choice = {sid: i for i, sid in enumerate(M_yoke_choice.columns)}
yoke_idx_recall = {sid: i for i, sid in enumerate(M_yoke_recall.columns)}

mFz = float(FREE_CHOICE_MEAN_Z)
vFz = float(np.var(FREE_CHOICE_ZS, ddof=1))
nFz = int(len(FREE_CHOICE_ZS))

rng = np.random.default_rng(SEED)
accepted = []
tries = 0

while len(accepted) < N_ACCEPT and tries < MAX_TRIES:
    tries += 1

    chosen = [rng.choice(lst) for lst in bin_lists]

    y_choice_r = uppvec_from_full(R_choice_yoke, chosen, yoke_idx_choice)
    if y_choice_r.size < 5:
        continue

    y_choice_z = fisher(y_choice_r)
    myz = float(np.mean(y_choice_z))
    vyz = float(np.var(y_choice_z, ddof=1))
    nyz = int(len(y_choice_z))

    if not (np.isfinite(myz) and myz <= mFz + EPS_Z):
        continue

    _ = welch_p_z(myz, vyz, nyz, mFz, vFz, nFz)  # not used for gating

    if any(s not in yoke_idx_recall for s in chosen):
        continue
    y_recall_r = uppvec_from_full(R_recall_yoke, chosen, yoke_idx_recall)
    if y_recall_r.size == 0:
        continue

    accepted.append(float(np.mean(y_recall_r)))

print(f"Accepted {len(accepted)} Yoked samples after {tries} attempts (target {N_ACCEPT}, max {MAX_TRIES}).")

accepted = np.asarray(accepted, float)
if accepted.size == 0:
    raise RuntimeError("No accepted samples; cannot build null distribution.")

p_np = (np.sum(accepted >= FREE_RECALL_MEAN) + 1) / (accepted.size + 1)

print(f"Null mean={accepted.mean():.3f}, sd={accepted.std(ddof=1):.3f}, n={accepted.size}")
print(f"Non-parametric p (Yoked Recall ≥ Free Recall, choice-matched in z): {p_np:.4g}")

# ======================= PLOTS =======================
# (1) Permutation null vs Free Recall ISC
fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.hist(accepted, bins=50, density=True, alpha=0.85, color="green",
        edgecolor="black", linewidth=0.8)
xs = np.linspace(min(accepted.min(), FREE_RECALL_MEAN),
                 max(accepted.max(), FREE_RECALL_MEAN), 600)
ax.plot(xs, gaussian_kde(accepted)(xs), linewidth=2, label="KDE (perm)")
mu, sd = accepted.mean(), accepted.std(ddof=1)
ax.plot(xs, norm.pdf(xs, mu, sd), "--", linewidth=2, label="Normal approx")
ax.axvline(FREE_RECALL_MEAN, color="blue", linestyle="--", linewidth=2)
ax.text(FREE_RECALL_MEAN, ax.get_ylim()[1]*0.95, f"Free = {FREE_RECALL_MEAN:.3f}",
        color="blue", ha="right", va="top", rotation=90,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5))
ax.set_title("Yoked permutation null (choice-matched, z-gated) vs Free Recall ISC")
ax.set_xlabel("Mean pairwise Recall ISC (r)")
ax.set_ylabel("Density")
ax.margins(x=0.02, y=0.05)
ax.legend()
ax.grid(alpha=0.2, linestyle=":")
plt.tight_layout()
plt.show()

# (2) Recall ISC bar plot (Free / Yoked / Passive)
def mean_sem_of_upper(M):
    v = upper_r_values(M)
    if v.size == 0:
        return np.nan, np.nan
    vz    = fisher(v)
    mean_r = float(np.nanmean(v))
    sem_z  = float(np.nanstd(vz, ddof=1) / np.sqrt(len(vz)))
    sem_r  = delta_sem_r(mean_r, sem_z)
    return mean_r, sem_r

mF, sF = mean_sem_of_upper(M_free_recall)
mY, sY = mean_sem_of_upper(M_yoke_recall)
mP, sP = mean_sem_of_upper(M_pasv_recall)

labels = ["Free","Yoked","Passive"]
means  = [mF, mY, mP]
sems   = [sF, sY, sP]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

fig, ax = plt.subplots(figsize=(6.5, 4.2))
bars = ax.bar(labels, means, yerr=sems, capsize=6,
              edgecolor="black", linewidth=1.2, alpha=0.95, color=colors)

ax.set_ylabel("Mean pairwise Recall ISC (r)")
ax.set_title("Recall ISC across shared events (mean ± SEM)")

tops = [m + (s if np.isfinite(s) else 0) for m, s in zip(means, sems)]
top  = np.nanmax(tops) if len(tops) else 1.0
ax.set_ylim(0, top * 1.10 if np.isfinite(top) and top > 0 else 1.0)

for b, m in zip(bars, means):
    if np.isfinite(m):
        ax.text(b.get_x() + b.get_width()/2,
                b.get_height() + (0.01 * ax.get_ylim()[1]),
                f"{m:.3f}", ha="center", va="bottom", fontsize=9)

ax.grid(axis="y", alpha=0.2, linestyle=":")
plt.tight_layout()
plt.show()
