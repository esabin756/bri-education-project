import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from patsy import dmatrices

panel = pd.read_csv('Data/Panel/panel_500m_treated.csv')

# ── Add event study dummies ──
def add_event_vars(df, lags=[0,1,2,3], pres=[1,2]):
    df = df.copy()
    for l in lags:
        df[f'lag{l}'] = ((df['treat_500m']==1) & (df['rel_time']==l)).astype(float)
    for p in pres:
        df[f'pre{p}'] = ((df['treat_500m']==1) & (df['rel_time']==-p)).astype(float)
    return df

panel = add_event_vars(panel)

controls = 'log_gdp_pc_current_usd + population_total + percent_urban + birth_rate_crude_per_1000'

def run_did(df, outcome, label):
    df = df.dropna(subset=[outcome])
    print(f"\n{'='*55}")
    print(f"OUTCOME: {label}")
    print(f"{'='*55}")

    # ── Basic DiD ──
    formula_basic = f'{outcome} ~ did + {controls} + C(year) + C(Q("Country Code"))'
    # rename did col
    df = df.copy()
    df['did'] = df['treat_500m'] * df['post_500m']

    mod_basic = smf.ols(formula_basic, data=df).fit(
        cov_type='HC1'
    )
    c = mod_basic.params['did']
    se = mod_basic.bse['did']
    t = mod_basic.tvalues['did']
    p = mod_basic.pvalues['did']
    stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
    print(f"\n[1] Basic DiD — N={int(mod_basic.nobs)}")
    print(f"    Post x Treated: {c:+.3f}{stars}  SE={se:.3f}  t={t:.2f}  p={p:.3f}")

    # ── Dynamic DiD ──
    lag_terms = ' + '.join([f'lag{l}' for l in [0,1,2,3]] + [f'pre{p}' for p in [1,2]])
    formula_dyn = f'{outcome} ~ {lag_terms} + {controls} + C(year) + C(Q("Country Code"))'

    # build design matrices so we can align cluster groups with rows used in the model
    y, X = dmatrices(formula_dyn, data=df, return_type='dataframe')
    groups = df.loc[X.index, 'Country Code'].astype('category').cat.codes
    mod_dyn = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': groups})

    print(f"\n[2] Dynamic DiD — N={int(mod_dyn.nobs)}")
    print(f"    {'Period':<10} {'Coef':>8} {'SE':>8} {'t':>7} {'p':>7} {'':>4}")
    print(f"    {'-'*45}")
    for pp, col in [(-2,'pre2'),(-1,'pre1'),(0,'lag0'),(1,'lag1'),(2,'lag2'),(3,'lag3')]:
        c = mod_dyn.params[col]
        se = mod_dyn.bse[col]
        t = mod_dyn.tvalues[col]
        p = mod_dyn.pvalues[col]
        stars = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        lbl = f"Pre({abs(pp)})" if pp < 0 else f"Lag({pp})"
        print(f"    {lbl:<10} {c:>+8.3f} {se:>8.3f} {t:>7.2f} {p:>7.3f} {stars:>4}")

# ── Run for both outcomes ──
run_did(panel, 'primary_enroll_gross_pct',   'Primary Enrollment (%)')
run_did(panel, 'secondary_enroll_gross_pct', 'Secondary Enrollment (%)')