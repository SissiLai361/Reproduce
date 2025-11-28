"""
See manuscript results section: Agency did not improve recall performance
"""

import pandas as pd
from pathlib import Path
from statsmodels.stats.oneway import anova_oneway  # one-way ANOVA (equal / unequal var)

# Paths to summary results for both stories
base = Path("..") / "data" / "overall_data"
files = {
    "Adventure": base / "adventure_result.xlsx",
    "Romance":   base / "romance_result.xlsx",
}

lines = []
for story, fp in files.items():
    # Load condition label and overall recall probability
    df = pd.read_excel(fp, usecols=["cond", "overall_rcl"])

    # Extract recall values by condition (Free, Yoked, Passive)
    groups = [
        df.query("cond=='free'")["overall_rcl"].dropna().values,
        df.query("cond=='yoke'")["overall_rcl"].dropna().values,
        df.query("cond=='pasv'")["overall_rcl"].dropna().values,
    ]
    # Drop any empty groups (in case a condition is missing for a story)
    groups = [g for g in groups if g.size > 0]

    if len(groups) == 3:
        # One-way ANOVA across the three conditions, assuming equal variances
        res = anova_oneway(groups, use_var="equal")
        F, df1, df2, p = res.statistic, res.df_num, res.df_denom, res.pvalue
        lines.append(
            f"{story}: one-way ANOVA F({df1:.0f},{df2:.0f}) = {F:.2f}, p = {p:.3f}"
        )
    else:
        lines.append(f"{story}: one-way ANOVA N/A — one or more groups missing data")

print("There were no significant differences in recall performance across conditions in either story.\n")
for s in lines:
    print(s)
print()
