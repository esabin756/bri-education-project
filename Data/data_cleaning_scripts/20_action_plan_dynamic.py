import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys, os

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
print("  Key insight: if effects take 5-6 years to materialize,")
print("  the 2016 Action Plan effects should appear 2021-2023")
print("  Panel runs to 2023 so we can just barely test this")
print("="*65)

# ── Build education investment post-2016 treatment ──
# Countries that crossed $10M in Chinese education investment
# AFTER 2016 specifically — these are the action plan recipients
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
print(f"Countries in full panel: {panel['Country Code'].nunique()}")

# Merge into panel
for df in [panel, ssa]:
    crossed_merge = crossed_post2016.copy()
    df_merged = df.merge(crossed_merge, on='Country Code', how='left')

panel = panel.merge(crossed_post2016, on='Country Code', how='left')
ssa   = ssa.merge(crossed_post2016,   on='Country Code', how='left')

panel['treat_post2016'] = panel['treat_year_post2016'].notna().astype(int)
ssa['treat_post2016']   = ssa['treat_year_post2016'].notna().astype(int)

print(f"Treated (post-2016 edu): {panel[panel['treat_post2016']==1]['Country Code'].nunique()}")
print(f"SSA treated:             {ssa[ssa['treat_post2016']==1]['Country Code'].nunique()}")

# ── Dynamic DiD from 2016 onward ──
# We restrict to 2012-2023 so we have 4 pre-treatment years
# and can observe up to 6 post-treatment years
def run_dynamic_post2016(df, outcome):
    # Restrict to 2012 onward for cleaner window around 2016
    df = df[df['year'] >= 2012].copy()
    df = df.dropna(subset=['Country Code','year','treat_post2016',
                            'treat_year_post2016',
                            'log_gdp_pc_current_usd','population_total',
                            'percent_urban','birth_rate_crude_per_1000',
                            outcome]).copy()

    # Relative time to treatment
    df['rel_time'] = df['year'] - df['treat_year_post2016']

    # Post-treatment lags 0-6
    for l in range(6):
        df[f'lag{l}'] = ((df['treat_post2016']==1) &
                          (df['rel_time']==l)).astype(float)

    # Pre-treatment periods 1-3
    # (we only have ~4 years pre-2016 so limit to 3 pre periods)
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

outcomes = [
    ('secondary_enroll_gross_pct', 'Secondary Enrollment Gross (%)'),
    ('tertiary_enroll_gross_pct',  'Tertiary Enrollment Gross (%)'),
    ('primary_enroll_gross_pct',   'Primary Enrollment Gross (%)'),
]

print("\n\n" + "#"*65)
print("  DYNAMIC DiD — POST-2016 EDUCATION INVESTMENT")
print("  Pre-periods: 2013-2015 | Post-periods: 2016-2022")
print("  Lag(5) = 2021, Lag(6) = 2022 — where we expect effects")
print("#"*65)

for ocol, olbl in outcomes:
    for sdf, slbl in [(panel,'Full Panel'),(ssa,'Sub-Saharan Africa')]:
        n_treated = sdf[sdf['treat_post2016']==1]['Country Code'].nunique()
        print(f"\n  {olbl} — {slbl} (N treated={n_treated})")
        print(f"  {'Period':<12} {'Coef':>8} {'SE':>8} {'p':>7} {'':>4}")
        print(f"  {'-'*40}")
        print(f"  {'[Pre-trend]':<12}")
        try:
            dyn, nobs, nc = run_dynamic_post2016(sdf, ocol)
            for _, row in dyn.iterrows():
                pp  = int(row['period'])
                lbl = f"Pre({abs(pp)})" if pp < 0 else f"Lag({pp})"
                sig = stars(row['p'])
                # Highlight the 5-6 year lags where we expect action plan effects
                marker = ' <- expect effect here' if pp == 5 else ''
                print(f"  {lbl:<12} {row['coef']:>+8.3f} "
                      f"{row['se']:>8.3f} {row['p']:>7.3f} {sig:>4}{marker}")
            print(f"  [N obs={nobs}, {nc} countries]")
        except Exception as e:
            print(f"  ERROR: {e}")

print("\n\nDone.")
log.close()