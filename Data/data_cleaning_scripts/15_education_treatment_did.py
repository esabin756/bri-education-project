import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys, os

_terminal = sys.stdout
class Tee:
    def __init__(self, t, l): self.terminal=t; self.logfile=l
    def write(self, o): self.terminal.write(o); self.terminal.flush(); self.logfile.write(o); self.logfile.flush()
    def flush(self): self.terminal.flush(); self.logfile.flush()

os.makedirs('Data/Regressions', exist_ok=True)
log = open('Data/Regressions/education_treatment_did_log.txt', 'w')
sys.stdout = Tee(_terminal, log)

panel = pd.read_csv('Data/Panel/panel_2023.csv')
aid   = pd.read_csv('Data/clean_aid_data/aiddata_clg_country_year.csv')

print("="*65)
print("  EDUCATION TREATMENT DiD — THRESHOLD ROBUSTNESS")
print(f"  Panel: {panel['Country Code'].nunique()} countries, "
      f"{panel['year'].min()}-{panel['year'].max()}")
print("="*65)

controls = ('log_gdp_pc_current_usd + population_total + '
            'percent_urban + birth_rate_crude_per_1000')
low_mid  = ['Low income','Lower middle income','Upper middle income']

full = panel.copy()
lm   = panel[panel['IncomeGroup'].isin(low_mid)].copy()
ssa  = panel[panel['Region']=='Sub-Saharan Africa'].copy()

def stars(p):
    return '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''

def build_edu_treatment(df, threshold_m):
    df = df.copy()
    edu_cumul = (aid.sort_values(['Country Code','year'])
                    .assign(cumul_edu=lambda x: x.groupby('Country Code')
                                                  ['usd_Education'].cumsum()))
    crossed = (edu_cumul[edu_cumul['cumul_edu'] >= threshold_m * 1e6]
                        .groupby('Country Code')['year']
                        .min().reset_index()
                        .rename(columns={'year': 'treat_year_edu_thresh'}))
    df = df.drop(columns=[c for c in df.columns
                           if any(x in c for x in
                                  ['treat_year_edu_thresh','treat_edu_t',
                                   'post_edu_t','rel_time_edu_t'])],
                 errors='ignore')
    df = df.merge(crossed, on='Country Code', how='left')
    df['treat_edu_t'] = df['treat_year_edu_thresh'].notna().astype(int)
    df['post_edu_t']  = ((df['treat_edu_t']==1) &
                          (df['year'] >= df['treat_year_edu_thresh'])).astype(int)
    return df, crossed

def run_basic_edu(df, outcome):
    df = df.dropna(subset=[outcome,'Country Code','year',
                            'treat_edu_t','post_edu_t',
                            'log_gdp_pc_current_usd','population_total',
                            'percent_urban','birth_rate_crude_per_1000']).copy()
    df['did'] = df['treat_edu_t'] * df['post_edu_t']
    mod = smf.ols(
        f'Q("{outcome}") ~ did + {controls} + '
        f'C(year) + C(Q("Country Code"))',
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['Country Code']})
    return (mod.params['did'], mod.bse['did'],
            mod.pvalues['did'], int(mod.nobs))

def run_dynamic_edu(df, outcome):
    df = df.dropna(subset=[outcome,'Country Code','year',
                            'treat_edu_t','treat_year_edu_thresh',
                            'log_gdp_pc_current_usd','population_total',
                            'percent_urban','birth_rate_crude_per_1000']).copy()
    df['rel_time_edu_t'] = df['year'] - df['treat_year_edu_thresh']
    for l in range(7):
        df[f'lag{l}'] = ((df['treat_edu_t']==1) &
                          (df['rel_time_edu_t']==l)).astype(float)
    for p in [1,2,3,4]:
        df[f'pre{p}'] = ((df['treat_edu_t']==1) &
                          (df['rel_time_edu_t']==-p)).astype(float)
    lag_terms = ' + '.join(
        [f'lag{l}' for l in range(7)] + ['pre1','pre2','pre3','pre4']
    )
    mod = smf.ols(
        f'Q("{outcome}") ~ {lag_terms} + {controls} + '
        f'C(year) + C(Q("Country Code"))',
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['Country Code']})
    results = []
    for pp, col in [(-4,'pre4'),(-3,'pre3'),(-2,'pre2'),(-1,'pre1'),
                    (0,'lag0'),(1,'lag1'),(2,'lag2'),(3,'lag3'),
                    (4,'lag4'),(5,'lag5'),(6,'lag6')]:
        results.append({'period': pp, 'coef': mod.params[col],
                        'se': mod.bse[col], 'p': mod.pvalues[col]})
    return pd.DataFrame(results)

# ── Outcomes — updated to use gross gender enrollment ──
outcomes = [
    ('primary_enroll_gross_pct',   'Primary Enrollment Gross (%)'),
    ('primary_gross_female',       'Primary Gross Female (%)'),
    ('primary_gross_male',         'Primary Gross Male (%)'),
    ('secondary_enroll_gross_pct', 'Secondary Enrollment Gross (%)'),
    ('secondary_gross_female',     'Secondary Gross Female (%)'),
    ('secondary_gross_male',       'Secondary Gross Male (%)'),
    ('secondary_gender_gap_gross', 'Secondary Gender Gap Gross (pp)'),
    ('tertiary_enroll_gross_pct',  'Tertiary Enrollment Gross (%)'),
    ('tertiary_gross_female_y',    'Tertiary Gross Female (%)'),
    ('tertiary_gross_male_y',      'Tertiary Gross Male (%)'),
    ('tertiary_gender_gap_gross',  'Tertiary Gender Gap Gross (pp)'),
    ('avg_years_schooling',        'Avg Years of Schooling (adults)'),
]

thresholds = [10, 50, 100]
subsets_info = [
    ('Full Panel',          full),
    ('Low & Middle Income', lm),
    ('Sub-Saharan Africa',  ssa),
]

# ================================================================
# PART 1: THRESHOLD ROBUSTNESS — BASIC DiD
# ================================================================
print(f"\n\n{'#'*65}")
print(f"  PART 1: THRESHOLD ROBUSTNESS — BASIC DiD")
print(f"{'#'*65}")

for ocol, olbl in outcomes:
    print(f"\n{'#'*65}")
    print(f"  {olbl}")
    print(f"{'#'*65}")
    for slbl, sdf in subsets_info:
        print(f"\n  {slbl}")
        print(f"  {'Threshold':<12} {'N Treated':>10} {'Coef':>8} "
              f"{'SE':>8} {'p':>7} {'N obs':>7} {'':>4}")
        print(f"  {'-'*58}")
        for thresh in thresholds:
            df_t, crossed = build_edu_treatment(sdf, thresh)
            n_treated = df_t[df_t['treat_edu_t']==1]['Country Code'].nunique()
            try:
                c, se, p, n = run_basic_edu(df_t, ocol)
                sig = stars(p)
                print(f"  ${thresh}M{'':<8} {n_treated:>10} {c:>+8.3f} "
                      f"{se:>8.3f} {p:>7.3f} {n:>7} {sig:>4}")
            except Exception as e:
                print(f"  ${thresh}M{'':<8} {n_treated:>10} ERROR: {e}")

# ================================================================
# PART 2: DYNAMIC DiD — $50M THRESHOLD
# ================================================================
print(f"\n\n{'#'*65}")
print(f"  PART 2: DYNAMIC DiD — $50M EDUCATION THRESHOLD")
print(f"{'#'*65}")

key_dynamic = [
    ('primary_enroll_gross_pct',   'Primary Enrollment Gross',   full, 'Full Panel'),
    ('primary_enroll_gross_pct',   'Primary Enrollment Gross',   ssa,  'Sub-Saharan Africa'),
    ('secondary_enroll_gross_pct', 'Secondary Enrollment Gross', full, 'Full Panel'),
    ('secondary_enroll_gross_pct', 'Secondary Enrollment Gross', ssa,  'Sub-Saharan Africa'),
    ('secondary_gross_female',     'Secondary Gross Female',     full, 'Full Panel'),
    ('secondary_gross_female',     'Secondary Gross Female',     ssa,  'Sub-Saharan Africa'),
    ('secondary_gross_male',       'Secondary Gross Male',       full, 'Full Panel'),
    ('secondary_gross_male',       'Secondary Gross Male',       ssa,  'Sub-Saharan Africa'),
    ('secondary_gender_gap_gross', 'Secondary Gender Gap Gross', full, 'Full Panel'),
    ('tertiary_enroll_gross_pct',  'Tertiary Enrollment Gross',  full, 'Full Panel'),
    ('tertiary_enroll_gross_pct',  'Tertiary Enrollment Gross',  ssa,  'Sub-Saharan Africa'),
    ('avg_years_schooling',        'Avg Years of Schooling',     full, 'Full Panel'),
    ('avg_years_schooling',        'Avg Years of Schooling',     ssa,  'Sub-Saharan Africa'),
]

for ocol, olbl, sdf, slbl in key_dynamic:
    df_t, crossed = build_edu_treatment(sdf, 50)
    n_treated = df_t[df_t['treat_edu_t']==1]['Country Code'].nunique()
    print(f"\n  {olbl} — {slbl} (N treated={n_treated})")
    print(f"  {'Period':<10} {'Coef':>8} {'SE':>8} {'p':>7} {'':>4}")
    print(f"  {'-'*38}")
    try:
        dyn = run_dynamic_edu(df_t, ocol)
        for _, row in dyn.iterrows():
            lbl = (f"Pre({abs(int(row['period']))})"
                   if row['period'] < 0
                   else f"Lag({int(row['period'])})")
            sig = stars(row['p'])
            print(f"  {lbl:<10} {row['coef']:>+8.3f} "
                  f"{row['se']:>8.3f} {row['p']:>7.3f} {sig:>4}")
    except Exception as e:
        print(f"  ERROR: {e}")

# ================================================================
# PART 3: SSA DYNAMIC DiD — ALL THRESHOLDS
# ================================================================
print(f"\n\n{'#'*65}")
print(f"  PART 3: SSA DYNAMIC DiD — SECONDARY ENROLLMENT")
print(f"{'#'*65}")

for thresh in thresholds:
    df_t, crossed = build_edu_treatment(ssa, thresh)
    n_treated = df_t[df_t['treat_edu_t']==1]['Country Code'].nunique()
    print(f"\n  ${thresh}M threshold — SSA (N treated={n_treated})")
    print(f"  {'Period':<10} {'Coef':>8} {'SE':>8} {'p':>7} {'':>4}")
    print(f"  {'-'*38}")
    try:
        dyn = run_dynamic_edu(df_t, 'secondary_enroll_gross_pct')
        for _, row in dyn.iterrows():
            lbl = (f"Pre({abs(int(row['period']))})"
                   if row['period'] < 0
                   else f"Lag({int(row['period'])})")
            sig = stars(row['p'])
            print(f"  {lbl:<10} {row['coef']:>+8.3f} "
                  f"{row['se']:>8.3f} {row['p']:>7.3f} {sig:>4}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n\nDone.")
log.close()