import pandas as pd
import numpy as np
import os
from scipy import stats as scipy_stats
import sys

_terminal = sys.stdout
log = open('Data/Regressions/summary_statistics_log.txt', 'w')
class Tee:
    def __init__(self, t, l): self.t=t; self.l=l
    def write(self, o): self.t.write(o); self.t.flush(); self.l.write(o); self.l.flush()
    def flush(self): self.t.flush(); self.l.flush()
sys.stdout = Tee(_terminal, log)

panel = pd.read_csv('Data/Panel/panel_2023.csv')

vars_info = {
    'secondary_enroll_gross_pct':  'Secondary Enrollment Gross (%)',
    'secondary_gross_female':      'Secondary Gross Female (%)',
    'secondary_gross_male':        'Secondary Gross Male (%)',
    'secondary_gender_gap_gross':  'Secondary Gender Gap (F-M, pp)',
    'primary_enroll_gross_pct':    'Primary Enrollment Gross (%)',
    'primary_gross_female':        'Primary Gross Female (%)',
    'primary_gross_male':          'Primary Gross Male (%)',
    'tertiary_enroll_gross_pct':   'Tertiary Enrollment Gross (%)',
    'tertiary_gross_female_y':     'Tertiary Gross Female (%)',
    'tertiary_gross_male_y':       'Tertiary Gross Male (%)',
    'avg_years_schooling':         'Avg Years of Schooling (adults)',
    'female_emp_ratio':            'Female Employment Ratio (%)',
    'youth_emp_ratio':             'Youth Employment Ratio (%)',
    'unemployment_pct':            'Unemployment Rate (%)',
    'gdp_growth':                  'GDP Growth (Annual %)',
    'gni_per_capita':              'GNI per Capita (USD)',
    'poverty_3day':                'Poverty Headcount $3/day (%)',
    'log_gdp_pc_current_usd':      'Log GDP per Capita',
    'percent_urban':               'Urban Population (%)',
    'birth_rate_crude_per_1000':   'Birth Rate (per 1,000)',
    'log1p_usd_const_2023':        'Log BRI Investment ($M)',
}

def summary_stats(df, label):
    rows = []
    for col, name in vars_info.items():
        if col not in df.columns:
            continue
        s = df[col].dropna()
        rows.append({
            'Variable':  name,
            'N':         len(s),
            'Mean':      s.mean(),
            'SD':        s.std(),
            'Min':       s.min(),
            'Max':       s.max(),
            'Missing %': df[col].isna().mean() * 100,
        })
    return pd.DataFrame(rows)

groups = [
    (panel,                                                              'Full Panel'),
    (panel[panel['treat_500m']==1],                                      'Treated ($500M+)'),
    (panel[panel['treat_500m']==0],                                      'Never Treated'),
    (panel[panel['Region']=='Sub-Saharan Africa'],                       'Sub-Saharan Africa'),
    (panel[(panel['Region']=='Sub-Saharan Africa') & (panel['treat_500m']==1)], 'SSA Treated'),
    (panel[(panel['Region']=='Sub-Saharan Africa') & (panel['treat_500m']==0)], 'SSA Never Treated'),
]

print("="*70)
print("  SUMMARY STATISTICS")
print("="*70)

for df, label in groups:
    n_countries = df['Country Code'].nunique()
    n_obs       = len(df)
    print(f"\n{'#'*70}")
    print(f"  {label}  |  {n_countries} countries  |  {n_obs:,} obs")
    print(f"{'#'*70}")
    print(f"  {'Variable':<35} {'N':>6} {'Mean':>8} {'SD':>8} "
          f"{'Min':>8} {'Max':>8} {'Miss%':>7}")
    print(f"  {'-'*82}")
    stats = summary_stats(df, label)
    for _, row in stats.iterrows():
        print(f"  {row['Variable']:<35} {int(row['N']):>6} "
              f"{row['Mean']:>8.2f} {row['SD']:>8.2f} "
              f"{row['Min']:>8.2f} {row['Max']:>8.2f} "
              f"{row['Missing %']:>6.1f}%")

# Treatment timing
print(f"\n\n{'#'*70}")
print(f"  TREATMENT TIMING")
print(f"{'#'*70}")
treat_years = (panel[panel['treat_500m']==1]
               .groupby('Country Code')['treat_year'].first().dropna())
print(f"\n  Total treated:           {len(treat_years)}")
print(f"  Never treated:           {panel[panel['treat_500m']==0]['Country Code'].nunique()}")
print(f"  Median treatment year:   {treat_years.median():.0f}")
print(f"  Mean treatment year:     {treat_years.mean():.1f}")

has_sec = panel.dropna(subset=['secondary_enroll_gross_pct'])
print(f"\n  Treated with sec data:   {has_sec[has_sec['treat_500m']==1]['Country Code'].nunique()}")
print(f"  Missing sec data:")
no_sec = panel.groupby('Country Code')['secondary_enroll_gross_pct'].apply(lambda x: x.isna().all())
treated_missing = [c for c in no_sec[no_sec].index
                   if panel[panel['Country Code']==c]['treat_500m'].iloc[0]==1]
print(f"  {treated_missing}")

# SSA balance check
print(f"\n\n{'#'*70}")
print(f"  SSA PRE-TREATMENT BALANCE CHECK")
print(f"{'#'*70}")
ssa = panel[panel['Region']=='Sub-Saharan Africa'].copy()
ssa_pre = []
for country, grp in ssa.groupby('Country Code'):
    treat_yr = grp['treat_year'].iloc[0]
    pre = grp[grp['year'] < treat_yr] if pd.notna(treat_yr) else grp
    ssa_pre.append(pre)
ssa_pre   = pd.concat(ssa_pre)
ssa_pre_t = ssa_pre[ssa_pre['treat_500m']==1]
ssa_pre_c = ssa_pre[ssa_pre['treat_500m']==0]

balance_vars = {
    'secondary_enroll_gross_pct': 'Secondary Enrollment Gross (%)',
    'secondary_gross_female':     'Secondary Gross Female (%)',
    'secondary_gross_male':       'Secondary Gross Male (%)',
    'avg_years_schooling':        'Avg Years of Schooling',
    'log_gdp_pc_current_usd':     'Log GDP per Capita',
    'female_emp_ratio':           'Female Employment Ratio (%)',
    'percent_urban':              'Urban Population (%)',
    'birth_rate_crude_per_1000':  'Birth Rate (per 1,000)',
    'gdp_growth':                 'GDP Growth (%)',
}

print(f"\n  {'Variable':<35} {'Treated':>10} {'Control':>10} {'Diff':>10} {'p-val':>8}")
print(f"  {'-'*75}")
for col, name in balance_vars.items():
    if col not in ssa_pre.columns:
        continue
    t_vals = ssa_pre_t[col].dropna()
    c_vals = ssa_pre_c[col].dropna()
    if len(t_vals) < 5 or len(c_vals) < 5:
        continue
    diff        = t_vals.mean() - c_vals.mean()
    _, pval     = scipy_stats.ttest_ind(t_vals, c_vals)
    sig = '***' if pval<0.01 else '**' if pval<0.05 else '*' if pval<0.1 else ''
    print(f"  {name:<35} {t_vals.mean():>10.2f} {c_vals.mean():>10.2f} "
          f"{diff:>+10.2f} {pval:>7.3f}{sig}")

os.makedirs('Data/Regressions', exist_ok=True)
with pd.ExcelWriter('Data/Regressions/summary_statistics.xlsx', engine='openpyxl') as writer:
    for df, label in groups:
        summary_stats(df, label).round(3).to_excel(
            writer, sheet_name=label[:31], index=False)

with open('Data/Regressions/summary_statistics_clean.txt', 'w') as f:
    f.write("SUMMARY STATISTICS — CLEAN VERSION FOR LATEX\n")
    f.write("="*70 + "\n\n")
    
    for df, label in groups:
        n_countries = df['Country Code'].nunique()
        n_obs = len(df)
        f.write(f"{label}  |  {n_countries} countries  |  {n_obs:,} obs\n")
        f.write("-"*60 + "\n")
        stats = summary_stats(df, label)
        f.write(f"{'Variable':<35} {'Mean':>8} {'SD':>8} {'Miss%':>7}\n")
        for _, row in stats.iterrows():
            f.write(f"  {row['Variable']:<33} {row['Mean']:>8.2f} "
                    f"{row['SD']:>8.2f} {row['Missing %']:>6.1f}%\n")
        f.write("\n")

print("\n\nSaved: Data/Regressions/summary_statistics.xlsx")

print("Saved: Data/Regressions/summary_statistics_clean.txt")

print("Done.")
log.close()