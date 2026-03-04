import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys

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

log = open('Data/Regressions/income_regressions_log.txt', 'w')
sys.stdout = Tee(_terminal, log)

# ── Load and merge ──
panel  = pd.read_csv('Data/Panel/panel_with_income.csv')
income = pd.read_csv('Data/Panel/income_panel.csv')

panel = panel.merge(
    income[['Country Code','year','gni_per_capita','gdp_growth','poverty_3day']],
    on=['Country Code','year'], how='left'
)

print(f"Panel after merge: {panel.shape}")
print(f"GNI coverage:      {panel['gni_per_capita'].notna().mean()*100:.1f}%")
print(f"GDP growth:        {panel['gdp_growth'].notna().mean()*100:.1f}%")
print(f"Poverty coverage:  {panel['poverty_3day'].notna().mean()*100:.1f}%")

controls = 'log_gdp_pc_current_usd + population_total + percent_urban + birth_rate_crude_per_1000'
low_mid  = ['Low income','Lower middle income','Upper middle income']

outcomes = [
    ('gni_per_capita', 'GNI per Capita (current USD)'),
    ('gdp_growth',     'GDP Growth (annual %)'),
    ('poverty_3day',   'Poverty Headcount $3/day (%)'),
]

def run_did(df, outcome, label, subset_label):
    cols_needed = [outcome, 'Country Code', 'year',
                   'log_gdp_pc_current_usd','population_total',
                   'percent_urban','birth_rate_crude_per_1000']
    df = df.dropna(subset=cols_needed).copy()
    df['did'] = df['treat_500m'] * df['post_500m']

    print(f"\n{'='*60}")
    print(f"  {subset_label} — {label}")
    print(f"  N countries={df['Country Code'].nunique()}, N obs={len(df)}")
    print(f"{'='*60}")

    # Basic DiD
    mod = smf.ols(
        f'{outcome} ~ did + {controls} + C(year) + C(Q("Country Code"))',
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['Country Code']})

    c  = mod.params['did']
    se = mod.bse['did']
    t  = mod.tvalues['did']
    p  = mod.pvalues['did']
    sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
    print(f"\n[1] Basic DiD — N={int(mod.nobs)}")
    print(f"    Post x Treated: {c:+.3f}{sig}  SE={se:.3f}  t={t:.2f}  p={p:.3f}")

    # Dynamic DiD
    for l in [0,1,2,3,4,5]:
        df[f'lag{l}'] = ((df['treat_500m']==1) & (df['rel_time']==l)).astype(float)
    for p_ in [1,2]:
        df[f'pre{p_}'] = ((df['treat_500m']==1) & (df['rel_time']==-p_)).astype(float)

    lag_terms = ' + '.join([f'lag{l}' for l in range(6)] + ['pre1','pre2'])
    mod_dyn = smf.ols(
        f'{outcome} ~ {lag_terms} + {controls} + C(year) + C(Q("Country Code"))',
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['Country Code']})

    print(f"\n[2] Dynamic DiD — N={int(mod_dyn.nobs)}")
    print(f"    {'Period':<10} {'Coef':>10} {'SE':>10} {'t':>7} {'p':>7}")
    print(f"    {'-'*48}")
    for pp, col in [(-2,'pre2'),(-1,'pre1'),(0,'lag0'),(1,'lag1'),
                    (2,'lag2'),(3,'lag3'),(4,'lag4'),(5,'lag5')]:
        c_  = mod_dyn.params[col]
        se_ = mod_dyn.bse[col]
        t_  = mod_dyn.tvalues[col]
        p_  = mod_dyn.pvalues[col]
        sig_= '***' if p_<0.01 else '**' if p_<0.05 else '*' if p_<0.1 else ''
        lbl = f"Pre({abs(pp)})" if pp<0 else f"Lag({pp})"
        print(f"    {lbl:<10} {c_:>+10.2f} {se_:>10.2f} {t_:>7.2f} {p_:>7.3f} {sig_:>4}")

# ── Define subsets ──
full  = panel.copy()
lm    = panel[panel['IncomeGroup'].isin(low_mid)].copy()
ssa   = panel[panel['Region']=='Sub-Saharan Africa'].copy()

subsets = [
    (full, 'Full Panel'),
    (lm,   'Low & Middle Income'),
    (ssa,  'Sub-Saharan Africa'),
]

print("\n" + "="*60)
print("  INCOME & GROWTH OUTCOMES — DiD REGRESSIONS")
print("  Treatment: $500M cumulative Chinese investment")
print("="*60)

for ocol, olbl in outcomes:
    print(f"\n\n{'#'*60}")
    print(f"  {olbl}")
    print(f"{'#'*60}")
    for sdf, slbl in subsets:
        try:
            run_did(sdf, ocol, olbl, slbl)
        except Exception as e:
            print(f"  ERROR {slbl}: {e}")

print("\n\nDone.")
log.close()