import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys, os, json

log = open('Data/Regressions/ssa_mixed_controls_log.txt', 'w')
_t = sys.stdout
class Tee:
    def __init__(self,t,l): self.t=t; self.l=l
    def write(self,o): self.t.write(o); self.t.flush(); self.l.write(o); self.l.flush()
    def flush(self): self.t.flush(); self.l.flush()
sys.stdout = Tee(_t, log)

panel = pd.read_csv('Data/Panel/panel_2023.csv')
controls = ('log_gdp_pc_current_usd + population_total + '
            'percent_urban + birth_rate_crude_per_1000')

ssa_treated = panel[(panel['Region']=='Sub-Saharan Africa') &
                     (panel['treat_500m']==1)].copy()
all_never   = panel[panel['treat_500m']==0].copy()
mixed       = pd.concat([ssa_treated, all_never])

def stars(p):
    return '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''

print("="*70)
print("  ROBUSTNESS: SSA TREATED + ALL NEVER-TREATED CONTROLS")
print(f"  SSA treated: {ssa_treated['Country Code'].nunique()}")
print(f"  Never-treated controls: {all_never['Country Code'].nunique()}")
print("="*70)

outcomes = [
    ('secondary_enroll_gross_pct', 'Secondary Enrollment Gross'),
    ('primary_enroll_gross_pct',   'Primary Enrollment Gross'),
    ('tertiary_enroll_gross_pct',  'Tertiary Enrollment Gross'),
    ('gdp_growth',                 'GDP Growth'),
    ('female_emp_ratio',           'Female Employment'),
]

print(f"\n{'#'*70}")
print(f"  BASIC DiD")
print(f"  {'Outcome':<30} {'Coef':>8} {'SE':>8} {'p':>7} {'N':>7} {'':>4}")
print(f"  {'-'*62}")

for ocol, olbl in outcomes:
    cols = [ocol,'Country Code','year','treat_500m','post_500m',
            'log_gdp_pc_current_usd','population_total',
            'percent_urban','birth_rate_crude_per_1000']
    df = mixed.dropna(subset=[c for c in cols
                               if c in mixed.columns]).copy()
    df['did'] = df['treat_500m'] * df['post_500m']
    try:
        mod = smf.ols(
            f'{ocol} ~ did + {controls} + '
            f'C(year) + C(Q("Country Code"))',
            data=df
        ).fit(cov_type='cluster',
              cov_kwds={'groups': df['Country Code']})
        c, se, p = mod.params['did'], mod.bse['did'], mod.pvalues['did']
        print(f"  {olbl:<30} {c:>+8.3f} {se:>8.3f} "
              f"{p:>7.3f} {int(mod.nobs):>7} {stars(p):>4}")
    except Exception as e:
        print(f"  {olbl:<30} ERROR: {e}")

def run_dynamic(df, outcome, label):
    df = df.copy()
    df = df.dropna(subset=[outcome,'Country Code','year',
                            'treat_500m','treat_year',
                            'log_gdp_pc_current_usd','population_total',
                            'percent_urban',
                            'birth_rate_crude_per_1000']).copy()
    df['rel_time'] = df['year'] - df['treat_year']
    for l in range(7):
        df[f'lag{l}'] = ((df['treat_500m']==1) &
                          (df['rel_time']==l)).astype(float)
    for p in range(1, 5):
        df[f'pre{p}'] = ((df['treat_500m']==1) &
                          (df['rel_time']==-p)).astype(float)
    lag_terms = ' + '.join(
        [f'lag{l}' for l in range(7)] +
        [f'pre{p}' for p in range(1, 5)]
    )
    mod = smf.ols(
        f'{outcome} ~ {lag_terms} + {controls} + '
        f'C(year) + C(Q("Country Code"))',
        data=df
    ).fit(cov_type='cluster',
          cov_kwds={'groups': df['Country Code']})

    print(f"\n{'#'*70}")
    print(f"  DYNAMIC DiD — {label}")
    print(f"  N obs={int(mod.nobs)}, "
          f"{df['Country Code'].nunique()} countries")
    print(f"{'#'*70}")
    print(f"  {'Period':<10} {'Coef':>8} {'SE':>8} {'p':>7} {'':>4}")
    print(f"  {'-'*38}")

    results = []
    for pp in range(-4, 7):
        col = f'pre{abs(pp)}' if pp < 0 else f'lag{pp}'
        if pp == -1:
            print(f"  {'Pre(1)':<10} {'0.000':>8} {'(norm)':>8}")
            results.append({'Period':'Pre(1)','Coef':0.0,
                            'SE':None,'p':None})
            continue
        lbl = f"Pre({abs(pp)})" if pp < 0 else f"Lag({pp})"
        c   = mod.params[col]
        se  = mod.bse[col]
        p   = mod.pvalues[col]
        sig = stars(p)
        print(f"  {lbl:<10} {c:>+8.3f} {se:>8.3f} "
              f"{p:>7.3f} {sig:>4}")
        results.append({'Period':lbl,'Coef':round(c,3),
                        'SE':round(se,3),'p':round(p,3)})
    return results

sec = run_dynamic(mixed, 'secondary_enroll_gross_pct',
                  'Secondary Enrollment Gross (%)')
pri = run_dynamic(mixed, 'primary_enroll_gross_pct',
                  'Primary Enrollment Gross (%)')
ter = run_dynamic(mixed, 'tertiary_enroll_gross_pct',
                  'Tertiary Enrollment Gross (%)')

os.makedirs('Data/Tables', exist_ok=True)
with open('Data/Tables/ssa_mixed_dynamic_all.json','w') as f:
    json.dump({'secondary':sec,'primary':pri,'tertiary':ter}, f, indent=2)
print("\nSaved: Data/Tables/ssa_mixed_dynamic_all.json")
print("\nDone.")
log.close()