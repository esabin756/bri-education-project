import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys, os

os.makedirs('Data/Regressions', exist_ok=True)
log = open('Data/Regressions/action_plan_dynamic_log.txt', 'w')
_t = sys.stdout
class Tee:
    def __init__(self,t,l): self.t=t; self.l=l
    def write(self,o): self.t.write(o); self.t.flush(); self.l.write(o); self.l.flush()
    def flush(self): self.t.flush(); self.l.flush()
sys.stdout = Tee(_t, log)

panel = pd.read_csv('Data/Panel/panel_2023.csv')
aid   = pd.read_csv('Data/clean_aid_data/aiddata_clg_country_year.csv')

controls = ('log_gdp_pc_current_usd + population_total + '
            'percent_urban + birth_rate_crude_per_1000')
ssa = panel[panel['Region']=='Sub-Saharan Africa'].copy()

def stars(p):
    return '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''

print("="*65)
print("  ACTION PLAN DYNAMIC DiD")
print(f"  Panel: {panel['Country Code'].nunique()} countries, "
      f"{panel['year'].min()}-{panel['year'].max()}")
print("  Post-2016 education investment treatment")
print("  Lag(5) = 2021, Lag(6) = 2022 — where we expect effects")
print("="*65)

# ── Build post-2016 education treatment ──
aid_post2016 = aid[aid['year'] >= 2016].copy()
edu_post2016 = (aid_post2016.sort_values(['Country Code','year'])
                .assign(cumul_edu_post16=lambda x: x.groupby('Country Code')
                                                    ['usd_Education'].cumsum()))
crossed_post2016 = (edu_post2016[edu_post2016['cumul_edu_post16'] >= 10e6]
                    .groupby('Country Code')['year']
                    .min().reset_index()
                    .rename(columns={'year': 'treat_year_post2016'}))

print(f"\nCountries crossing $10M edu investment post-2016: "
      f"{len(crossed_post2016)}")

# Drop existing columns if present to avoid duplicates
for df_name, df in [('panel', panel), ('ssa', ssa)]:
    drop_cols = [c for c in df.columns if 'treat_year_post2016' in c
                 or 'treat_post2016' in c]
    df.drop(columns=drop_cols, errors='ignore', inplace=True)

panel = panel.merge(crossed_post2016, on='Country Code', how='left')
ssa   = ssa.merge(crossed_post2016,   on='Country Code', how='left')

panel['treat_post2016'] = panel['treat_year_post2016'].notna().astype(int)
ssa['treat_post2016']   = ssa['treat_year_post2016'].notna().astype(int)

print(f"Treated post-2016 edu (full): "
      f"{panel[panel['treat_post2016']==1]['Country Code'].nunique()}")
print(f"Treated post-2016 edu (SSA):  "
      f"{ssa[ssa['treat_post2016']==1]['Country Code'].nunique()}")

# ── Dynamic DiD ──
def run_dynamic_post2016(df, outcome):
    df = df[df['year'] >= 2012].copy()
    df = df.dropna(subset=[outcome, 'Country Code','year',
                            'treat_post2016','treat_year_post2016',
                            'log_gdp_pc_current_usd','population_total',
                            'percent_urban','birth_rate_crude_per_1000']).copy()
    df['rel_time'] = df['year'] - df['treat_year_post2016']
    for l in range(6):
        df[f'lag{l}'] = ((df['treat_post2016']==1) &
                          (df['rel_time']==l)).astype(float)
    for p in range(1, 4):
        df[f'pre{p}'] = ((df['treat_post2016']==1) &
                          (df['rel_time']==-p)).astype(float)
    lag_terms = ' + '.join(
        [f'lag{l}' for l in range(6)] + ['pre1','pre2','pre3']
    )
    mod = smf.ols(
        f'Q("{outcome}") ~ {lag_terms} + {controls} + '
        f'C(year) + C(Q("Country Code"))',
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['Country Code']})
    rows = []
    for pp in range(-3, 6):
        col = f'pre{abs(pp)}' if pp < 0 else f'lag{pp}'
        if pp == -1:
            rows.append({'period': pp, 'coef': 0.0, 'se': 0.0, 'p': 1.0})
            continue
        rows.append({
            'period': pp,
            'coef':   mod.params.get(col, np.nan),
            'se':     mod.bse.get(col, np.nan),
            'p':      mod.pvalues.get(col, np.nan),
        })
    return pd.DataFrame(rows), int(mod.nobs), df['Country Code'].nunique()

# ── Updated outcomes ──
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

print("\n\n" + "#"*65)
print("  DYNAMIC DiD — POST-2016 EDUCATION INVESTMENT")
print("  Pre-periods: 2013-2015 | Post-periods: 2016-2022")
print("#"*65)

for ocol, olbl in outcomes:
    for sdf, slbl in [(panel,'Full Panel'),(ssa,'Sub-Saharan Africa')]:
        n_treated = sdf[sdf['treat_post2016']==1]['Country Code'].nunique()
        print(f"\n  {olbl} — {slbl} (N treated={n_treated})")
        print(f"  {'Period':<12} {'Coef':>8} {'SE':>8} {'p':>7} {'':>4}")
        print(f"  {'-'*40}")
        try:
            dyn, nobs, nc = run_dynamic_post2016(sdf, ocol)
            for _, row in dyn.iterrows():
                pp  = int(row['period'])
                lbl = f"Pre({abs(pp)})" if pp < 0 else f"Lag({pp})"
                sig = stars(row['p'])
                marker = ' <- expect effect here' if pp == 5 else ''
                print(f"  {lbl:<12} {row['coef']:>+8.3f} "
                      f"{row['se']:>8.3f} {row['p']:>7.3f} "
                      f"{sig:>4}{marker}")
            print(f"  [N obs={nobs}, {nc} countries]")
        except Exception as e:
            print(f"  ERROR: {e}")

print("\n\nDone.")
sys.stdout = _t
log.close()