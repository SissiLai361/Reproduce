"""
See manuscript results section: Pairwise recall similarity among Free participants (64 shared events)
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

# ---------- Paths (Romance, cross-platform) ----------
base_overall = Path("..") / "data" / "overall_data"
base_indiv   = Path("..") / "data" / "individual_data"

MAP_XLSX    = base_overall / "monthy_map.xlsx"
FREE_RECALL = base_indiv / "recall_prob" / "romance" / "1_free" / "recall_allsub.xlsx"
FREE_EVTS   = base_indiv / "rating"      / "romance" / "1_free"  # sub10_4010_events.xlsx

PATTERN_FR  = r"sub\d+_4\d+"   # Free IDs, e.g. sub10_4010


def shared_events(path: Path):
    """Event labels shared across paths (Converge == 'Y')."""
    m = pd.read_excel(path)
    m["Converge"]  = m["Converge"].astype(str).str.strip().str.upper()
    m["event_lab"] = m["event_lab"].astype(str).str.strip()
    return (m.loc[m["Converge"].eq("Y"), "event_lab"]
             .drop_duplicates()
             .tolist())


def subj_pos(ev_path: Path, shared, prefer=("old_seg", "event_lab", "event", "event_id")):
    """Map event_lab -> first index for one subject's *_events.xlsx."""
    df = pd.read_excel(ev_path)
    df.columns = df.columns.map(str)
    low = {c.lower(): c for c in df.columns}

    # Pick a reasonable event id column
    col = next((low[p.lower()] for p in prefer if p.lower() in low), None)
    if col is None:
        S = set(map(str, shared))
        col = max(
            df.columns,
            key=lambda c: pd.Series(df[c].astype(str).str.strip()).isin(S).sum()
        )

    ev = df[col].astype(str).str.strip()
    first = ~ev.duplicated(keep="first")
    return {e: i for i, e in zip(np.where(first)[0], ev[first])}


def build_free_matrix():
    """Rows: shared events, Cols: Free subjects (recall probabilities)."""
    shared = shared_events(MAP_XLSX)

    R = pd.read_excel(FREE_RECALL)
    R.columns = R.columns.map(str)

    subs = [c for c in R.columns if re.fullmatch(PATTERN_FR, c)]
    if not subs:
        raise ValueError("No Free subject columns matching pattern found.")

    R = R[subs].apply(pd.to_numeric, errors="coerce")

    cols = {}
    for sid in subs:
        ev_path = FREE_EVTS / f"{sid}_events.xlsx"
        pos = subj_pos(ev_path, shared)
        cols[sid] = [
            float(R.at[pos[e], sid]) if (e in pos and pd.notna(R.at[pos[e], sid])) else np.nan
            for e in shared
        ]

    return pd.DataFrame(cols, index=shared).clip(0, 1)


# ---------- Main ----------
M = build_free_matrix()   # rows: 64 shared events, cols: Free subjects
subs = M.columns.tolist()

print("Pairwise Pearson r between Free subjects over the 64 shared events:\n")
for i in range(len(subs) - 1):
    xi = M[subs[i]].to_numpy(float)
    for j in range(i + 1, len(subs)):
        xj = M[subs[j]].to_numpy(float)
        mask = np.isfinite(xi) & np.isfinite(xj)
        if mask.sum() >= 2:
            r = pearsonr(xi[mask], xj[mask])[0]
        else:
            r = np.nan
        print(f"{subs[i]} vs {subs[j]}: r = {r:.3f}")
print()
