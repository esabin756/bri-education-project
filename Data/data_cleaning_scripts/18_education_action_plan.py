import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys, os

os.makedirs('Data/Regressions', exist_ok=True)
log = open('Data/Regressions/education_action_plan_log.txt', 'w')
_t = sys.stdout
class Tee:
    def __init__(self,t,l): self.t=t; self.l=l
    def write(self,o): self.t.write(o); self.t.flush(); self.l.write(o); self.l.flush()
    def flush(self): self.t.flush(); self.l.flush()
sys.stdout = Tee(_t, log)

panel = pd.read_csv('Data/Panel/panel_2023.csv')

controls = ('log_gdp_pc_current_usd + population_total + '
            'percent_urban + birth_rate_crude_per_1000')
low_mid = ['Low income','Lower middle income','Upper middle income']
ssa     = panel[panel['Region']=='Sub-Saharan Africa'].copy()

def stars(p):
    return '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''

print("="*65)
print("  2016 EDUCATION ACTION PLAN — DiD ANALYSIS")
print(f"  Panel: {panel['Country Code'].nunique()} countries, "
      f"{panel['year'].min()}-{panel['year'].max()}")
print("  Did BRI education outcomes change after 2016?")
print("="*65)

# ── Action plan indicators ──
for df in [panel, ssa]:
    df['post_2016']        = (df['year'] >= 2016).astype(int)
    df['treat_x_post2016'] = df['treat_500m'] * df['post_2016']

# ── Updated outcomes including gross gender and avg years ──
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
    ('avg_years_schooling',        'Avg Years of Schooling (adults)'),
    ('female_emp_ratio',           'Female Employment Ratio (%)'),
    ('gdp_growth',                 'GDP Growth (%)'),
]

subsets = [
    (panel,                                   'Full Panel'),
    (panel[panel['IncomeGroup'].isin(low_mid)].copy(), 'Low & Middle Income'),
    (ssa,                                     'Sub-Saharan Africa'),
]

# ================================================================
# PART 1: POST-2016 INTERACTION
# ================================================================
print("\n\n" + "#"*65)
print("  PART 1: POST-2016 INTERACTION")
print("  Coefficient of interest: Treated x Post-2016")
print("#"*65)

for ocol, olbl in outcomes:
    print(f"\n  {olbl}")
    print(f"  {'Subset':<25} {'Coef':>8} {'SE':>8} {'p':>7} {'N':>7} {'':>4}")
    print(f"  {'-'*58}")
    for sdf, slbl in subsets:
        cols = [ocol,'Country Code','year','treat_500m',
                'post_2016','treat_x_post2016',
                'log_gdp_pc_current_usd','population_total',
                'percent_urban','birth_rate_crude_per_1000']
        df = sdf.dropna(subset=[c for c in cols
                                 if c in sdf.columns]).copy()
        try:
            mod = smf.ols(
                f'Q("{ocol}") ~ treat_x_post2016 + treat_500m + '
                f'post_2016 + {controls} + '
                f'C(year) + C(Q("Country Code"))',
                data=df
            ).fit(cov_type='cluster',
                  cov_kwds={'groups': df['Country Code']})
            c   = mod.params['treat_x_post2016']
            se  = mod.bse['treat_x_post2016']
            p   = mod.pvalues['treat_x_post2016']
            sig = stars(p)
            print(f"  {slbl:<25} {c:>+8.3f} {se:>8.3f} "
                  f"{p:>7.3f} {int(mod.nobs):>7} {sig:>4}")
        except Exception as e:
            print(f"  {slbl:<25} ERROR: {e}")

# ================================================================
# PART 2: PRE vs POST 2016 SPLIT
# ================================================================
print("\n\n" + "#"*65)
print("  PART 2: SEPARATE DiD — PRE-2016 vs POST-2016 PERIOD")
print("#"*65)

for ocol, olbl in outcomes:
    print(f"\n  {olbl}")
    print(f"  {'Period':<20} {'Subset':<25} {'Coef':>8} "
          f"{'SE':>8} {'p':>7} {'N':>7} {'':>4}")
    print(f"  {'-'*75}")
    for period_label, yr_min, yr_max in [
        ('Pre-2016  (2000-2015)', 2000, 2015),
        ('Post-2016 (2016-2023)', 2016, 2023),
    ]:
        for sdf, slbl in subsets:
            cols = [ocol,'Country Code','year','treat_500m','post_500m',
                    'log_gdp_pc_current_usd','population_total',
                    'percent_urban','birth_rate_crude_per_1000']
            df = sdf[(sdf['year']>=yr_min) &
                     (sdf['year']<=yr_max)].dropna(
                subset=[c for c in cols if c in sdf.columns]).copy()
            df['did'] = df['treat_500m'] * df['post_500m']
            try:
                mod = smf.ols(
                    f'Q("{ocol}") ~ did + {controls} + '
                    f'C(year) + C(Q("Country Code"))',
                    data=df
                ).fit(cov_type='cluster',
                      cov_kwds={'groups': df['Country Code']})
                c   = mod.params['did']
                se  = mod.bse['did']
                p   = mod.pvalues['did']
                sig = stars(p)
                print(f"  {period_label:<20} {slbl:<25} {c:>+8.3f} "
                      f"{se:>8.3f} {p:>7.3f} {int(mod.nobs):>7} {sig:>4}")
            except Exception as e:
                print(f"  {period_label:<20} {slbl:<25} ERROR: {e}")

# ================================================================
# PART 3: EDUCATION INVESTMENT POST-2016
# ================================================================
print("\n\n" + "#"*65)
print("  PART 3: EDUCATION-SPECIFIC INVESTMENT POST-2016")
print("#"*65)

for df in [panel, ssa]:
    df['treat_edu_post2016'] = df['treat_edu'] * df['post_2016']

for ocol, olbl in outcomes:
    print(f"\n  {olbl}")
    print(f"  {'Subset':<25} {'Coef':>8} {'SE':>8} {'p':>7} {'N':>7} {'':>4}")
    print(f"  {'-'*58}")
    for sdf, slbl in subsets:
        if 'treat_edu_post2016' not in sdf.columns:
            sdf = sdf.copy()
            sdf['treat_edu_post2016'] = sdf['treat_edu'] * sdf['post_2016']
        cols = [ocol,'Country Code','year','treat_edu',
                'post_edu','treat_edu_post2016',
                'log_gdp_pc_current_usd','population_total',
                'percent_urban','birth_rate_crude_per_1000']
        df = sdf.dropna(subset=[c for c in cols
                                  if c in sdf.columns]).copy()
        df['did_edu'] = df['treat_edu'] * df['post_edu']
        try:
            mod = smf.ols(
                f'Q("{ocol}") ~ did_edu + treat_edu_post2016 + '
                f'{controls} + C(year) + C(Q("Country Code"))',
                data=df
            ).fit(cov_type='cluster',
                  cov_kwds={'groups': df['Country Code']})
            c   = mod.params['treat_edu_post2016']
            se  = mod.bse['treat_edu_post2016']
            p   = mod.pvalues['treat_edu_post2016']
            sig = stars(p)
            print(f"  {slbl:<25} {c:>+8.3f} {se:>8.3f} "
                  f"{p:>7.3f} {int(mod.nobs):>7} {sig:>4}")
        except Exception as e:
            print(f"  {slbl:<25} ERROR: {e}")

# ================================================================
# PART 4: CUTOFF SENSITIVITY — 2014 2015 2016 2017
# ================================================================
print("\n\n" + "#"*65)
print("  PART 4: CUTOFF SENSITIVITY — 2014 THROUGH 2017")
print("#"*65)

key_outcomes = [
    ('secondary_enroll_gross_pct', 'Secondary Enrollment Gross'),
    ('avg_years_schooling',        'Avg Years of Schooling'),
    ('gdp_growth',                 'GDP Growth'),
]

for ocol, olbl in key_outcomes:
    print(f"\n  {olbl}")
    print(f"  {'Cutoff':<10} {'Subset':<25} {'Coef':>8} "
          f"{'SE':>8} {'p':>7} {'N':>7} {'':>4}")
    print(f"  {'-'*68}")
    for cutoff in [2014, 2015, 2016, 2017]:
        for sdf, slbl in subsets:
            df = sdf.copy()
            df['post_cut']     = (df['year'] >= cutoff).astype(int)
            df['treat_x_post'] = df['treat_500m'] * df['post_cut']
            cols = [ocol,'Country Code','year','treat_500m',
                    'post_cut','treat_x_post',
                    'log_gdp_pc_current_usd','population_total',
                    'percent_urban','birth_rate_crude_per_1000']
            df = df.dropna(subset=[c for c in cols
                                    if c in df.columns]).copy()
            try:
                mod = smf.ols(
                    f'Q("{ocol}") ~ treat_x_post + treat_500m + '
                    f'post_cut + {controls} + '
                    f'C(year) + C(Q("Country Code"))',
                    data=df
                ).fit(cov_type='cluster',
                      cov_kwds={'groups': df['Country Code']})
                c   = mod.params['treat_x_post']
                se  = mod.bse['treat_x_post']
                p   = mod.pvalues['treat_x_post']
                sig = stars(p)
                print(f"  {cutoff:<10} {slbl:<25} {c:>+8.3f} "
                      f"{se:>8.3f} {p:>7.3f} {int(mod.nobs):>7} {sig:>4}")
            except Exception as e:
                print(f"  {cutoff:<10} {slbl:<25} ERROR: {e}")

print("\n\nDone.")
sys.stdout = _t
log.close()