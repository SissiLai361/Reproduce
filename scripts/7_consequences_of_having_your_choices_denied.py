"""
See manuscript results section: Consequences of having your choices denied
"""

import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind, pearsonr

# ======================= Paths =======================
base_overall = Path("..") / "data" / "overall_data"
base_indiv   = Path("..") / "data" / "individual_data"

ADV_RESULT_XLS = base_overall / "adventure_result.xlsx"
ROM_RESULT_XLS = base_overall / "romance_result.xlsx"

ADV_FREE_RECALL = base_indiv / "recall_prob" / "adventure" / "1_free" / "recall_allsub.xlsx"
ROM_FREE_RECALL = base_indiv / "recall_prob" / "romance"   / "1_free" / "recall_allsub.xlsx"

ADV_YOKE_SHEET = "yoke45"
ROM_YOKE_SHEET = "yoke53"


# ======================= Small helpers =======================
def parse_num_vec(s):
    """
    Parse comma-separated strings like '1,0,1' into a float array.
    Returns empty array for NaN/empty cells.
    """
    if pd.isna(s):
        return np.array([], float)
    if isinstance(s, (int, float)):
        return np.array([float(s)], float)

    txt = str(s).strip()
    if not txt:
        return np.array([], float)

    vals = []
    for p in txt.split(","):
        p = p.strip()
        if not p:
            continue
        try:
            vals.append(float(p))
        except ValueError:
            continue
    return np.asarray(vals, float)


def yoke_to_free_id(yoke_sid: str, free_cols: list[str]) -> str | None:
    """
    Map yoke ID like 'to23_sub5001' to free ID like 'sub23_10xx' or 'sub23_40xx':
      - extract path number from 'toXX'
      - pick free column whose name starts with 'subXX_'
    """
    m = re.fullmatch(r"to(\d+)_sub\d+", str(yoke_sid).strip())
    if not m:
        return None
    prefix = f"sub{m.group(1)}_"
    matches = [c for c in free_cols if str(c).startswith(prefix)]
    if not matches:
        return None
    if len(matches) > 1:
        print(f"Warning: multiple free IDs for {yoke_sid}: {matches}; using {matches[0]}")
    return matches[0]


# ======================= Per-yoked-subject summary =======================
def process_story(story_label: str,
                  result_xls: Path,
                  yoke_sheet: str,
                  free_recall_xls: Path) -> pd.DataFrame:
    """
    Build a yoked-subject summary table for one story.

      index: yoke_subj  (e.g., 'to23_sub5001')

      columns:
        free_subj
        pct_granted                 # proportion of choices granted
        yoke_granted_mean           # Yoked recall for granted events
        yoke_denied_mean            # Yoked recall for denied events
        free_granted_mean           # Free recall at those granted events
        free_denied_mean            # Free recall at those denied events
        tendency_r                  # r(wn, yoke recall) across choice events
    """
    df_res = pd.read_excel(result_xls, sheet_name=yoke_sheet)
    df_res.columns = [str(c).strip().lower() for c in df_res.columns]

    for c in ("subj", "wn", "rcl", "inds"):
        if c not in df_res.columns:
            raise ValueError(f"Missing column '{c}' in {result_xls} sheet {yoke_sheet}")

    # Free recall matrix: rows = events, cols = free subjects
    R_free = pd.read_excel(free_recall_xls)
    R_free.columns = [str(c).strip() for c in R_free.columns]
    free_cols = list(R_free.columns)

    rows_out = []

    def safe_mean(x):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        return float(x.mean()) if x.size else np.nan

    def safe_corr(x, y):
        """Within-subject correlation r(wn, yoke recall)."""
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]
        if x.size < 3 or np.all(x == x[0]) or np.all(y == y[0]):
            return np.nan
        r, _ = pearsonr(x, y)
        return float(r)

    for _, row in df_res.iterrows():
        yoke_sid = str(row["subj"]).strip()
        if not re.fullmatch(r"to\d+_sub\d+", yoke_sid):
            continue

        wn_vec   = parse_num_vec(row["wn"])
        rcl_vec  = parse_num_vec(row["rcl"])
        inds_vec = parse_num_vec(row["inds"]).astype(int)

        # Align lengths across wn, recall, event indices
        L = min(len(wn_vec), len(rcl_vec), len(inds_vec))
        if L == 0:
            continue
        wn_vec, rcl_vec, inds_vec = wn_vec[:L], rcl_vec[:L], inds_vec[:L]

        # Keep only valid binary wn
        mask_valid = np.isin(wn_vec, [0.0, 1.0]) & np.isfinite(wn_vec)
        if not np.any(mask_valid):
            continue
        wn_vec, rcl_vec, inds_vec = wn_vec[mask_valid], rcl_vec[mask_valid], inds_vec[mask_valid]

        # Map yoked subject to its free counterpart
        free_sid = yoke_to_free_id(yoke_sid, free_cols)
        if free_sid is None or free_sid not in R_free.columns:
            print(f"{story_label}: cannot map yoke {yoke_sid} to free; skipping")
            continue

        free_full = pd.to_numeric(R_free[free_sid], errors="coerce").to_numpy()

        # Use inds as indices into the free subject's recall vector
        valid_idx = (inds_vec >= 0) & (inds_vec < len(free_full))
        if not np.any(valid_idx):
            print(f"{story_label}: {yoke_sid}/{free_sid} has no valid indices; skipping")
            continue

        wn_vec   = wn_vec[valid_idx]
        rcl_vec  = rcl_vec[valid_idx]
        inds_vec = inds_vec[valid_idx]
        free_vec = free_full[inds_vec]

        granted = (wn_vec == 1.0)
        denied  = (wn_vec == 0.0)

        n_grant = int(granted.sum())
        n_deny  = int(denied.sum())
        n_total = n_grant + n_deny
        if n_total == 0:
            continue

        yoke_grant_vals = rcl_vec[granted]
        yoke_deny_vals  = rcl_vec[denied]
        free_grant_vals = free_vec[granted]
        free_deny_vals  = free_vec[denied]

        rows_out.append({
            "yoke_subj":         yoke_sid,
            "free_subj":         free_sid,
            "pct_granted":       n_grant / n_total,
            "yoke_granted_mean": safe_mean(yoke_grant_vals),
            "yoke_denied_mean":  safe_mean(yoke_deny_vals),
            "free_granted_mean": safe_mean(free_grant_vals),
            "free_denied_mean":  safe_mean(free_deny_vals),
            "tendency_r":        safe_corr(wn_vec, rcl_vec),
        })

    if not rows_out:
        print(f"{story_label}: no valid yoke subjects after filtering.")
        return pd.DataFrame()

    return pd.DataFrame(rows_out).set_index("yoke_subj")


# ======================= Build summaries for both stories =======================
adv_df = process_story("Adventure", ADV_RESULT_XLS, ADV_YOKE_SHEET, ADV_FREE_RECALL)
rom_df = process_story("Romance",   ROM_RESULT_XLS, ROM_YOKE_SHEET, ROM_FREE_RECALL)


# ======================= 7.1 Denied vs Granted =======================
print(
    "7.1 Having one’s agency denied can have unique memory effects at local choice "
    "events: one’s memory for the denied choice events is selectively reduced "
    "compared to its choice-granted counterparts in the Free condition.\n"
)

def s7_1_denied_effect(df, label):
    """
    7.1: Compare denied-event recall:
      Yoked denied vs Free denied (two-sample t-test).
    """
    if df is None or df.empty:
        print(f"{label}: no valid data for 7.1\n")
        return

    yoke_den = df["yoke_denied_mean"].dropna().astype(float)
    free_den = df["free_denied_mean"].dropna().astype(float)
    if len(yoke_den) < 2 or len(free_den) < 2:
        print(f"{label}: insufficient subjects for denied-event comparison\n")
        return

    t_stat, p_val = ttest_ind(yoke_den, free_den, equal_var=True)
    df_eff = len(yoke_den) + len(free_den) - 2

    print(f"{label} — Denied choice events (Yoked vs Free counterparts)")
    print(f"  Yoked denied:  n={len(yoke_den)}, mean={yoke_den.mean():.3f}, sd={yoke_den.std(ddof=1):.3f}")
    print(f"  Free denied:   n={len(free_den)}, mean={free_den.mean():.3f}, sd={free_den.std(ddof=1):.3f}")
    print(f"  t({df_eff}) = {t_stat:.3f}, p = {p_val:.3f}\n")


s7_1_denied_effect(adv_df, "Adventure")
s7_1_denied_effect(rom_df, "Romance")


# ============================================================
# 7.2 %Granted predicting tendency to forget denied events
# ============================================================
print(
    "7.2.2 Higher percentage of choices granted predicted greater individual "
    "tendency to forget the choice-denied events.\n"
)

def run_7_2(story_label, df):
    """
    7.2: Correlate % choices granted with tendency_r
         (individual tendency to recall granted events more than denied ones).
    """
    if df is None or df.empty:
        print(f"{story_label}: missing data\n")
        return

    pg   = df["pct_granted"].astype(float).to_numpy()
    tend = df["tendency_r"].astype(float).to_numpy()
    mask = np.isfinite(pg) & np.isfinite(tend)
    pg, tend = pg[mask], tend[mask]
    if len(pg) < 3:
        print(f"{story_label}: insufficient data\n")
        return

    r, p = pearsonr(pg, tend)
    print(f"{story_label}: r({len(pg)-2}) = {r:.3f}, p = {p:.3f}\n")


run_7_2("Adventure", adv_df)
run_7_2("Romance",   rom_df)
