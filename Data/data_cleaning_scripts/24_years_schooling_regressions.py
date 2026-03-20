import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys, os

log = open('Data/Regressions/years_schooling_log.txt', 'w')
_t = sys.stdout
class Tee:
    def __init__(self,t,l): self.t=t; self.l=l
    def write(self,o): self.t.write(o); self.t.flush(); self.l.write(o); self.l.flush()
    def flush(self): self.t.flush(); self.l.flush()
sys.stdout = Tee(_t, log)

# ── Load years of schooling ──
edyr = pd.read_csv('Data/Raw_WBD/average_years_schooling.csv')
edyr = edyr.rename(columns={
    'Code':         'Country Code',
    'Year':         'year',
    'Both genders': 'avg_years_schooling'
})
edyr = edyr[edyr['Country Code'].notna()].copy()
edyr = edyr[['Country Code','year','avg_years_schooling']]
edyr = edyr[edyr['year'].between(2000, 2023)]

print("="*65)
print("  AVERAGE YEARS OF SCHOOLING — DiD REGRESSIONS")
print(f"  Source: Our World in Data / Lee-Lee Dataset")
print(f"  Path: Data/Raw_WBD/average_years_schooling.csv")
print(f"  Countries: {edyr['Country Code'].nunique()}")
print(f"  Years: {edyr['year'].min()} - {edyr['year'].max()}")
print(f"  Missing: {edyr['avg_years_schooling'].isna().mean()*100:.1f}%")
print("="*65)

# ── Merge into main panel ──
panel = pd.read_csv('Data/Panel/panel_2023.csv')
panel = panel.merge(edyr, on=['Country Code','year'], how='left')

print(f"\nAfter merge:")
print(f"  Panel obs with schooling data: "
      f"{panel['avg_years_schooling'].notna().sum():,}")
print(f"  Missing in merged panel: "
      f"{panel['avg_years_schooling'].isna().mean()*100:.1f}%")

controls = ('log_gdp_pc_current_usd + population_total + '
            'percent_urban + birth_rate_crude_per_1000')
low_mid  = ['Low income','Lower middle income','Upper middle income']
ssa      = panel[panel['Region']=='Sub-Saharan Africa'].copy()
lm       = panel[panel['IncomeGroup'].isin(low_mid)].copy()

def stars(p):
    return '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''

subsets = [
    (panel, 'Full Panel'),
    (lm,    'Low & Middle Income'),
    (ssa,   'Sub-Saharan Africa'),
]

# ================================================================
# PART 1: BASIC DiD
# ================================================================
print(f"\n\n{'#'*65}")
print(f"  PART 1: BASIC DiD — Average Years of Schooling")
print(f"  Treatment: $500M cumulative BRI investment")
print(f"{'#'*65}")

print(f"\n  {'Subset':<25} {'Coef':>8} {'SE':>8} {'p':>7} "
      f"{'N':>7} {'Countries':>10} {'':>4}")
print(f"  {'-'*70}")

for sdf, slbl in subsets:
    cols = ['avg_years_schooling','Country Code','year',
            'treat_500m','post_500m',
            'log_gdp_pc_current_usd','population_total',
            'percent_urban','birth_rate_crude_per_1000']
    df = sdf.dropna(subset=[c for c in cols
                             if c in sdf.columns]).copy()
    df['did'] = df['treat_500m'] * df['post_500m']
    try:
        mod = smf.ols(
            f'avg_years_schooling ~ did + {controls} + '
            f'C(year) + C(Q("Country Code"))',
            data=df
        ).fit(cov_type='cluster',
              cov_kwds={'groups': df['Country Code']})
        c   = mod.params['did']
        se  = mod.bse['did']
        p   = mod.pvalues['did']
        sig = stars(p)
        print(f"  {slbl:<25} {c:>+8.4f} {se:>8.4f} {p:>7.3f} "
              f"{int(mod.nobs):>7} {df['Country Code'].nunique():>10} {sig:>4}")
    except Exception as e:
        print(f"  {slbl:<25} ERROR: {e}")

# ================================================================
# PART 2: DYNAMIC DiD WITH PRE-TRENDS
# ================================================================
print(f"\n\n{'#'*65}")
print(f"  PART 2: DYNAMIC DiD — Pre(4) through Lag(6)")
print(f"{'#'*65}")

def run_dynamic(df, outcome, n_lags=6, n_pre=4):
    df = df.copy()
    df = df.dropna(subset=['Country Code','year','treat_500m',
                            'treat_year','log_gdp_pc_current_usd',
                            'population_total','percent_urban',
                            'birth_rate_crude_per_1000',
                            outcome]).copy()
    df['rel_time'] = df['year'] - df['treat_year']
    for l in range(n_lags+1):
        df[f'lag{l}'] = ((df['treat_500m']==1) &
                          (df['rel_time']==l)).astype(float)
    for p in range(1, n_pre+1):
        df[f'pre{p}'] = ((df['treat_500m']==1) &
                          (df['rel_time']==-p)).astype(float)
    lag_terms = ' + '.join(
        [f'lag{l}' for l in range(n_lags+1)] +
        [f'pre{p}' for p in range(1, n_pre+1)]
    )
    mod = smf.ols(
        f'{outcome} ~ {lag_terms} + {controls} + '
        f'C(year) + C(Q("Country Code"))',
        data=df
    ).fit(cov_type='cluster',
          cov_kwds={'groups': df['Country Code']})
    rows = []
    for pp in range(-n_pre, n_lags+1):
        col = f'pre{abs(pp)}' if pp < 0 else f'lag{pp}'
        if pp == -1:
            rows.append({'period':pp,'coef':0.0,'se':0.0,'p':1.0})
            continue
        rows.append({
            'period': pp,
            'coef':   mod.params.get(col, np.nan),
            'se':     mod.bse.get(col, np.nan),
            'p':      mod.pvalues.get(col, np.nan),
        })
    return pd.DataFrame(rows), int(mod.nobs), df['Country Code'].nunique()

for sdf, slbl in subsets:
    n_treated = sdf[sdf['treat_500m']==1]['Country Code'].nunique()
    print(f"\n  {slbl} (N treated={n_treated})")
    print(f"  {'Period':<10} {'Coef':>8} {'SE':>8} {'p':>7} {'':>4}")
    print(f"  {'-'*38}")
    try:
        dyn, nobs, nc = run_dynamic(sdf, 'avg_years_schooling')
        for _, row in dyn.iterrows():
            pp  = int(row['period'])
            lbl = f"Pre({abs(pp)})" if pp < 0 else f"Lag({pp})"
            sig = stars(row['p'])
            print(f"  {lbl:<10} {row['coef']:>+8.4f} "
                  f"{row['se']:>8.4f} {row['p']:>7.3f} {sig:>4}")
        print(f"  [N obs={nobs}, {nc} countries]")
    except Exception as e:
        print(f"  ERROR: {e}")

# ================================================================
# PART 3: EDUCATION TREATMENT DiD
# ================================================================
print(f"\n\n{'#'*65}")
print(f"  PART 3: EDUCATION-SPECIFIC TREATMENT ($10M)")
print(f"{'#'*65}")

print(f"\n  {'Subset':<25} {'Coef':>8} {'SE':>8} {'p':>7} "
      f"{'N':>7} {'':>4}")
print(f"  {'-'*55}")

for sdf, slbl in subsets:
    cols = ['avg_years_schooling','Country Code','year',
            'treat_edu','post_edu',
            'log_gdp_pc_current_usd','population_total',
            'percent_urban','birth_rate_crude_per_1000']
    df = sdf.dropna(subset=[c for c in cols
                              if c in sdf.columns]).copy()
    df['did_edu'] = df['treat_edu'] * df['post_edu']
    try:
        mod = smf.ols(
            f'avg_years_schooling ~ did_edu + {controls} + '
            f'C(year) + C(Q("Country Code"))',
            data=df
        ).fit(cov_type='cluster',
              cov_kwds={'groups': df['Country Code']})
        c   = mod.params['did_edu']
        se  = mod.bse['did_edu']
        p   = mod.pvalues['did_edu']
        sig = stars(p)
        print(f"  {slbl:<25} {c:>+8.4f} {se:>8.4f} "
              f"{p:>7.3f} {int(mod.nobs):>7} {sig:>4}")
    except Exception as e:
        print(f"  {slbl:<25} ERROR: {e}")

# ================================================================
# PART 4: SECTOR DOSE-RESPONSE
# ================================================================
print(f"\n\n{'#'*65}")
print(f"  PART 4: SECTOR DOSE-RESPONSE")
print(f"{'#'*65}")

sector_vars = [
    ('log1p_usd_const_2023', 'Total BRI Investment'),
    ('log1p_Education',      'Education Investment'),
    ('log1p_Transport',      'Transport Investment'),
    ('log1p_Energy',         'Energy Investment'),
    ('log1p_Health',         'Health Investment'),
    ('log1p_Industry',       'Industry Investment'),
]

for sdf, slbl in [(panel,'Full Panel'),(ssa,'Sub-Saharan Africa')]:
    print(f"\n  {slbl}")
    print(f"  {'Sector':<25} {'Coef':>8} {'SE':>8} {'p':>7} {'':>4}")
    print(f"  {'-'*50}")
    cols_needed = ['avg_years_schooling','Country Code','year',
                   'log_gdp_pc_current_usd','population_total',
                   'percent_urban','birth_rate_crude_per_1000']
    df = sdf.dropna(subset=cols_needed).copy()
    for var, lbl in sector_vars:
        if var not in df.columns:
            continue
        try:
            mod = smf.ols(
                f'avg_years_schooling ~ {var} + {controls} + '
                f'C(year) + C(Q("Country Code"))',
                data=df
            ).fit(cov_type='cluster',
                  cov_kwds={'groups': df['Country Code']})
            c   = mod.params[var]
            se  = mod.bse[var]
            p   = mod.pvalues[var]
            sig = stars(p)
            print(f"  {lbl:<25} {c:>+8.4f} {se:>8.4f} "
                  f"{p:>7.3f} {sig:>4}")
        except Exception as e:
            print(f"  {lbl:<25} ERROR: {e}")

print("\n\nDone.")
# restore stdout before closing the log to avoid EOF flush error
sys.stdout = _t
log.close()