import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys, os

log = open('Data/Regressions/years_schooling_action_plan_log.txt', 'w')
_t = sys.stdout
class Tee:
    def __init__(self,t,l): self.t=t; self.l=l
    def write(self,o): self.t.write(o); self.t.flush(); self.l.write(o); self.l.flush()
    def flush(self): self.t.flush(); self.l.flush()
sys.stdout = Tee(_t, log)

panel = pd.read_csv('Data/Panel/panel_2023.csv')
edyr  = pd.read_csv('Data/Raw_WBD/average_years_schooling.csv')
edyr  = edyr.rename(columns={
    'Code': 'Country Code', 'Year': 'year',
    'Both genders': 'avg_years_schooling'
})
edyr  = edyr[edyr['Country Code'].notna()]
edyr  = edyr[['Country Code','year','avg_years_schooling']]
edyr  = edyr[edyr['year'].between(2000,2023)]
panel = panel.merge(edyr, on=['Country Code','year'], how='left')

controls = ('log_gdp_pc_current_usd + population_total + '
            'percent_urban + birth_rate_crude_per_1000')
ssa = panel[panel['Region']=='Sub-Saharan Africa'].copy()

def stars(p):
    return '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''

print("="*65)
print("  YEARS OF SCHOOLING — 2016 EDUCATION ACTION PLAN")
print("  Did years of schooling improve in BRI countries after 2016?")
print("="*65)

# ── Add post-2016 interaction ──
for df in [panel, ssa]:
    df['post_2016']        = (df['year'] >= 2016).astype(int)
    df['treat_x_post2016'] = df['treat_500m'] * df['post_2016']

subsets = [
    (panel, 'Full Panel'),
    (panel[panel['IncomeGroup'].isin(
        ['Low income','Lower middle income','Upper middle income'])].copy(),
     'Low & Middle Income'),
    (ssa,   'Sub-Saharan Africa'),
]

# ================================================================
# PART 1: POST-2016 INTERACTION
# ================================================================
print(f"\n{'#'*65}")
print(f"  PART 1: DID YEARS OF SCHOOLING CHANGE POST-2016")
print(f"  Coef of interest: Treated x Post-2016")
print(f"{'#'*65}")

for cutoff in [2016, 2017]:
    print(f"\n  Cutoff year: {cutoff}")
    print(f"  {'Subset':<25} {'Coef':>8} {'SE':>8} {'p':>7} "
          f"{'N':>7} {'':>4}")
    print(f"  {'-'*58}")
    for sdf, slbl in subsets:
        df = sdf.copy()
        df['post_cut']     = (df['year'] >= cutoff).astype(int)
        df['treat_x_post'] = df['treat_500m'] * df['post_cut']
        cols = ['avg_years_schooling','Country Code','year',
                'treat_500m','post_cut','treat_x_post',
                'log_gdp_pc_current_usd','population_total',
                'percent_urban','birth_rate_crude_per_1000']
        df = df.dropna(subset=[c for c in cols
                                if c in df.columns]).copy()
        try:
            mod = smf.ols(
                f'avg_years_schooling ~ treat_x_post + treat_500m + '
                f'post_cut + {controls} + '
                f'C(year) + C(Q("Country Code"))',
                data=df
            ).fit(cov_type='cluster',
                  cov_kwds={'groups': df['Country Code']})
            c   = mod.params['treat_x_post']
            se  = mod.bse['treat_x_post']
            p   = mod.pvalues['treat_x_post']
            sig = stars(p)
            print(f"  {slbl:<25} {c:>+8.4f} {se:>8.4f} "
                  f"{p:>7.3f} {int(mod.nobs):>7} {sig:>4}")
        except Exception as e:
            print(f"  {slbl:<25} ERROR: {e}")

# ================================================================
# PART 2: PRE vs POST 2016 SPLIT
# ================================================================
print(f"\n\n{'#'*65}")
print(f"  PART 2: SEPARATE DiD PRE vs POST 2016")
print(f"  Did the treatment effect on years of schooling change?")
print(f"{'#'*65}")

for period_lbl, yr_min, yr_max in [
    ('Pre-2016  (2000-2015)', 2000, 2015),
    ('Post-2016 (2016-2023)', 2016, 2023),
]:
    print(f"\n  {period_lbl}")
    print(f"  {'Subset':<25} {'Coef':>8} {'SE':>8} {'p':>7} "
          f"{'N':>7} {'':>4}")
    print(f"  {'-'*58}")
    for sdf, slbl in subsets:
        cols = ['avg_years_schooling','Country Code','year',
                'treat_500m','post_500m',
                'log_gdp_pc_current_usd','population_total',
                'percent_urban','birth_rate_crude_per_1000']
        df = sdf[(sdf['year']>=yr_min) &
                 (sdf['year']<=yr_max)].dropna(
            subset=[c for c in cols if c in sdf.columns]).copy()
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
            print(f"  {slbl:<25} {c:>+8.4f} {se:>8.4f} "
                  f"{p:>7.3f} {int(mod.nobs):>7} {sig:>4}")
        except Exception as e:
            print(f"  {slbl:<25} ERROR: {e}")

# ================================================================
# PART 3: DYNAMIC DiD POST-2016 EDUCATION INVESTMENT
# ================================================================
print(f"\n\n{'#'*65}")
print(f"  PART 3: DYNAMIC DiD — POST-2016 EDUCATION INVESTMENT")
print(f"  Does education investment after 2016 affect years of schooling?")
print(f"  Expecting effects at Lag(5)+ given slow-moving stock variable")
print(f"{'#'*65}")

aid = pd.read_csv('Data/clean_aid_data/aiddata_clg_country_year.csv')

# Build post-2016 education treatment
aid_post2016 = aid[aid['year'] >= 2016].copy()
edu_post2016 = (aid_post2016.sort_values(['Country Code','year'])
                .assign(cumul=lambda x: x.groupby('Country Code')
                                         ['usd_Education'].cumsum()))
crossed = (edu_post2016[edu_post2016['cumul'] >= 10e6]
           .groupby('Country Code')['year'].min().reset_index()
           .rename(columns={'year': 'treat_year_post2016'}))

for df in [panel, ssa]:
    df.drop(columns=[c for c in df.columns
                     if 'treat_year_post2016' in c], errors='ignore',
            inplace=True)

panel = panel.merge(crossed, on='Country Code', how='left')
ssa   = ssa.merge(crossed,   on='Country Code', how='left')

panel['treat_post2016'] = panel['treat_year_post2016'].notna().astype(int)
ssa['treat_post2016']   = ssa['treat_year_post2016'].notna().astype(int)

def run_dynamic_post2016(df, outcome):
    df = df[df['year'] >= 2012].copy()
    df = df.dropna(subset=['Country Code','year','treat_post2016',
                            'treat_year_post2016',
                            'log_gdp_pc_current_usd','population_total',
                            'percent_urban','birth_rate_crude_per_1000',
                            outcome]).copy()
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
        f'{outcome} ~ {lag_terms} + {controls} + '
        f'C(year) + C(Q("Country Code"))',
        data=df
    ).fit(cov_type='cluster',
          cov_kwds={'groups': df['Country Code']})
    rows = []
    for pp in range(-3, 6):
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

for sdf, slbl in [(panel,'Full Panel'),(ssa,'Sub-Saharan Africa')]:
    n_t = sdf[sdf['treat_post2016']==1]['Country Code'].nunique()
    print(f"\n  {slbl} (N treated={n_t})")
    print(f"  {'Period':<10} {'Coef':>8} {'SE':>8} {'p':>7} {'':>4}")
    print(f"  {'-'*38}")
    try:
        dyn, nobs, nc = run_dynamic_post2016(sdf, 'avg_years_schooling')
        for _, row in dyn.iterrows():
            pp  = int(row['period'])
            lbl = f"Pre({abs(pp)})" if pp < 0 else f"Lag({pp})"
            sig = stars(row['p'])
            marker = ' <- expect effect here' if pp == 5 else ''
            print(f"  {lbl:<10} {row['coef']:>+8.4f} "
                  f"{row['se']:>8.4f} {row['p']:>7.3f} "
                  f"{sig:>4}{marker}")
        print(f"  [N obs={nobs}, {nc} countries]")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n\nDone.")
# restore stdout before closing the log to avoid EOF flush error
sys.stdout = _t
log.close()