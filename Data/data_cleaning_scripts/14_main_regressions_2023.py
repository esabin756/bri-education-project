import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys
import os

_terminal = sys.stdout

class Tee:
    def __init__(self, terminal, logfile):
        self.terminal = terminal
        self.logfile  = logfile
    def write(self, obj):
        self.terminal.write(obj)
        self.terminal.flush()
        self.logfile.write(obj)
        self.logfile.flush()
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

os.makedirs('Data/Regressions', exist_ok=True)
log = open('Data/Regressions/main_regressions_2023_log.txt', 'w')
sys.stdout = Tee(_terminal, log)

panel = pd.read_csv('Data/Panel/panel_2023.csv')

print("="*65)
print("  MAIN DiD REGRESSIONS — EXTENDED PANEL 2000-2023")
print(f"  N countries={panel['Country Code'].nunique()}, "
      f"Years={panel['year'].min()}-{panel['year'].max()}")
print(f"  Treated={panel[panel['treat_500m']==1]['Country Code'].nunique()}, "
      f"Never-treated={panel[panel['treat_500m']==0]['Country Code'].nunique()}")
print("="*65)

controls  = ('log_gdp_pc_current_usd + population_total + '
             'percent_urban + birth_rate_crude_per_1000')
low_mid   = ['Low income','Lower middle income','Upper middle income']

full = panel.copy()
lm   = panel[panel['IncomeGroup'].isin(low_mid)].copy()
ssa  = panel[panel['Region']=='Sub-Saharan Africa'].copy()

# ── All outcomes ──
outcomes = [
    # Education — gross
    ('primary_enroll_gross_pct',   'Primary Enrollment Gross (%)'),
    ('secondary_enroll_gross_pct', 'Secondary Enrollment Gross (%)'),
    ('tertiary_enroll_gross_pct',  'Tertiary Enrollment Gross (%)'),
    # Education — net and gender
    ('secondary_net_pct',          'Secondary Enrollment Net (%)'),
    ('secondary_net_female',       'Secondary Net — Female (%)'),
    ('secondary_net_male',         'Secondary Net — Male (%)'),
    ('secondary_gender_gap',       'Secondary Gender Gap (F-M, pp)'),
    ('tertiary_gross_female',      'Tertiary Gross — Female (%)'),
    ('tertiary_gross_male',        'Tertiary Gross — Male (%)'),
    # Employment
    ('female_emp_ratio',           'Female Employment Ratio (%)'),
    ('youth_emp_ratio',            'Youth Employment Ratio (%)'),
    ('unemployment_pct',           'Unemployment Rate (%)'),
    ('industry_emp_pct',           'Industry Employment (%)'),
    # Income
    ('gdp_growth',                 'GDP Growth (%)'),
    ('gni_per_capita',             'GNI per Capita (USD)'),
]

def stars(p):
    return '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''

def run_basic(df, outcome):
    cols = [outcome,'Country Code','year','treat_500m','post_500m',
            'log_gdp_pc_current_usd','population_total',
            'percent_urban','birth_rate_crude_per_1000']
    df = df.dropna(subset=[c for c in cols if c in df.columns]).copy()
    df['did'] = df['treat_500m'] * df['post_500m']
    mod = smf.ols(
        f'{outcome} ~ did + {controls} + C(year) + C(Q("Country Code"))',
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['Country Code']})
    return (mod.params['did'], mod.bse['did'],
            mod.pvalues['did'], int(mod.nobs),
            df['Country Code'].nunique())

def run_dynamic(df, outcome):
    cols = [outcome,'Country Code','year','treat_500m','post_500m','rel_time',
            'log_gdp_pc_current_usd','population_total',
            'percent_urban','birth_rate_crude_per_1000']
    df = df.dropna(subset=[c for c in cols if c in df.columns
                           and c != 'rel_time']).copy()
    df['rel_time'] = df['year'] - df['treat_year']

    for l in range(7):  # lags 0-6 now
        df[f'lag{l}'] = ((df['treat_500m']==1) &
                          (df['rel_time']==l)).astype(float)
    for p in [1,2]:
        df[f'pre{p}'] = ((df['treat_500m']==1) &
                          (df['rel_time']==-p)).astype(float)

    lag_terms = ' + '.join([f'lag{l}' for l in range(7)] + ['pre1','pre2'])
    mod = smf.ols(
        f'{outcome} ~ {lag_terms} + {controls} + C(year) + C(Q("Country Code"))',
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['Country Code']})

    results = []
    for pp, col in [(-2,'pre2'),(-1,'pre1'),(0,'lag0'),(1,'lag1'),
                    (2,'lag2'),(3,'lag3'),(4,'lag4'),(5,'lag5'),(6,'lag6')]:
        results.append({
            'period': pp,
            'coef':   mod.params[col],
            'se':     mod.bse[col],
            'p':      mod.pvalues[col],
        })
    return pd.DataFrame(results)

# ── Run education treatment separately ──
def run_edu_treatment(df, outcome):
    cols = [outcome,'Country Code','year','treat_edu','post_edu',
            'log_gdp_pc_current_usd','population_total',
            'percent_urban','birth_rate_crude_per_1000']
    df = df.dropna(subset=[c for c in cols if c in df.columns]).copy()
    df['did_edu'] = df['treat_edu'] * df['post_edu']
    mod = smf.ols(
        f'{outcome} ~ did_edu + {controls} + C(year) + C(Q("Country Code"))',
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['Country Code']})
    return (mod.params['did_edu'], mod.bse['did_edu'],
            mod.pvalues['did_edu'], int(mod.nobs),
            df['Country Code'].nunique())

# ================================================================
# PART 1: BASIC DiD — ALL OUTCOMES, ALL SUBSETS
# ================================================================
print("\n\n" + "#"*65)
print("  PART 1: BASIC DiD — Post x Treated")
print("#"*65)

subsets = [
    (full, 'Full Panel'),
    (lm,   'Low & Middle Income'),
    (ssa,  'Sub-Saharan Africa'),
]

for ocol, olbl in outcomes:
    print(f"\n{'='*65}")
    print(f"  {olbl}")
    print(f"  {'Subset':<25} {'Coef':>8} {'SE':>8} {'t':>7} "
          f"{'p':>7} {'N':>7} {'':>4}")
    print(f"  {'-'*62}")
    for sdf, slbl in subsets:
        try:
            c, se, p, n, nc = run_basic(sdf, ocol)
            sig = stars(p)
            print(f"  {slbl:<25} {c:>+8.3f} {se:>8.3f} "
                  f"{c/se:>7.2f} {p:>7.3f} {n:>7} {sig:>4}")
        except Exception as e:
            print(f"  {slbl:<25} ERROR: {e}")

# ================================================================
# PART 2: DYNAMIC DiD — KEY OUTCOMES
# ================================================================
print("\n\n" + "#"*65)
print("  PART 2: DYNAMIC DiD — KEY OUTCOMES")
print("#"*65)

key_outcomes = [
    ('secondary_enroll_gross_pct', 'Secondary Enrollment Gross', full, 'Full Panel'),
    ('secondary_enroll_gross_pct', 'Secondary Enrollment Gross', ssa,  'Sub-Saharan Africa'),
    ('secondary_net_pct',          'Secondary Enrollment Net',   ssa,  'Sub-Saharan Africa'),
    ('secondary_net_female',       'Secondary Net Female',       ssa,  'Sub-Saharan Africa'),
    ('secondary_net_male',         'Secondary Net Male',         ssa,  'Sub-Saharan Africa'),
    ('female_emp_ratio',           'Female Employment',          full, 'Full Panel'),
    ('gdp_growth',                 'GDP Growth',                 ssa,  'Sub-Saharan Africa'),
]

for ocol, olbl, sdf, slbl in key_outcomes:
    print(f"\n  {olbl} — {slbl}")
    print(f"  {'Period':<10} {'Coef':>8} {'SE':>8} {'p':>7} {'':>4}")
    print(f"  {'-'*38}")
    try:
        dyn = run_dynamic(sdf, ocol)
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
# PART 3: EDUCATION-SPECIFIC TREATMENT
# ================================================================
print("\n\n" + "#"*65)
print("  PART 3: EDUCATION-SPECIFIC TREATMENT ($10M edu investment)")
print(f"  Treated: {panel[panel['treat_edu']==1]['Country Code'].nunique()} countries")
print("#"*65)

edu_outcomes = [
    ('primary_enroll_gross_pct',   'Primary Enrollment Gross (%)'),
    ('secondary_enroll_gross_pct', 'Secondary Enrollment Gross (%)'),
    ('tertiary_enroll_gross_pct',  'Tertiary Enrollment Gross (%)'),
    ('secondary_net_pct',          'Secondary Enrollment Net (%)'),
    ('secondary_net_female',       'Secondary Net Female (%)'),
    ('secondary_net_male',         'Secondary Net Male (%)'),
]

for ocol, olbl in edu_outcomes:
    print(f"\n  {olbl}")
    print(f"  {'Subset':<25} {'Coef':>8} {'SE':>8} {'p':>7} {'N':>7} {'':>4}")
    print(f"  {'-'*58}")
    for sdf, slbl in subsets:
        try:
            c, se, p, n, nc = run_edu_treatment(sdf, ocol)
            sig = stars(p)
            print(f"  {slbl:<25} {c:>+8.3f} {se:>8.3f} "
                  f"{p:>7.3f} {n:>7} {sig:>4}")
        except Exception as e:
            print(f"  {slbl:<25} ERROR: {e}")

# ================================================================
# PART 4: SECTOR DOSE-RESPONSE — NEW DATA
# ================================================================
print("\n\n" + "#"*65)
print("  PART 4: SECTOR DOSE-RESPONSE — CLG AIDDATA")
print("#"*65)

sector_vars = [
    ('log1p_Education', 'Education Investment'),
    ('log1p_Transport', 'Transport Investment'),
    ('log1p_Energy',    'Energy Investment'),
    ('log1p_Health',    'Health Investment'),
    ('log1p_Industry',  'Industry Investment'),
]

sector_outcomes = [
    ('secondary_enroll_gross_pct', 'Secondary Enrollment Gross'),
    ('secondary_net_pct',          'Secondary Enrollment Net'),
    ('secondary_net_female',       'Secondary Net Female'),
    ('secondary_net_male',         'Secondary Net Male'),
    ('female_emp_ratio',           'Female Employment'),
    ('gdp_growth',                 'GDP Growth'),
]

for ocol, olbl in sector_outcomes:
    print(f"\n  {olbl}")
    for sdf, slbl in [(full,'Full Panel'),(ssa,'Sub-Saharan Africa')]:
        print(f"\n    {slbl}")
        print(f"    {'Sector':<25} {'Coef':>8} {'SE':>8} {'p':>7} {'':>4}")
        print(f"    {'-'*52}")
        cols_needed = [ocol,'Country Code','year',
                       'log_gdp_pc_current_usd','population_total',
                       'percent_urban','birth_rate_crude_per_1000']
        df2 = sdf.dropna(subset=cols_needed).copy()
        for svar, slabel in sector_vars:
            try:
                mod = smf.ols(
                    f'{ocol} ~ {svar} + {controls} + '
                    f'C(year) + C(Q("Country Code"))',
                    data=df2
                ).fit(cov_type='cluster',
                      cov_kwds={'groups': df2['Country Code']})
                c  = mod.params[svar]
                se = mod.bse[svar]
                p  = mod.pvalues[svar]
                sig = stars(p)
                print(f"    {slabel:<25} {c:>+8.4f} {se:>8.4f} "
                      f"{p:>7.3f} {sig:>4}")
            except Exception as e:
                print(f"    {slabel:<25} ERROR: {e}")

print("\n\nDone.")
log.close()