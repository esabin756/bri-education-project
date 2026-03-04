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

log = open('Data/Regressions/employment_regressions_log.txt', 'w')
sys.stdout = Tee(_terminal, log)

# ── Load and merge ──
panel = pd.read_csv('Data/Panel/panel_with_income.csv')
emp   = pd.read_csv('Data/Panel/employment_panel.csv')

panel = panel.merge(emp[['Country Code','year',
                          'youth_emp_ratio','female_emp_ratio',
                          'industry_emp_pct','unemployment_pct',
                          'youth_unemp_pct']],
                    on=['Country Code','year'], how='left')

print(f"Panel after merge: {panel.shape}")
print(f"Employment coverage: {panel['youth_emp_ratio'].notna().mean()*100:.1f}%")

controls  = 'log_gdp_pc_current_usd + population_total + percent_urban + birth_rate_crude_per_1000'
low_mid   = ['Low income','Lower middle income','Upper middle income']

outcomes = [
    ('youth_emp_ratio',   'Youth Employment Ratio 15-24 (%)'),
    ('female_emp_ratio',  'Female Employment Ratio 15+ (%)'),
    ('industry_emp_pct',  'Industry Employment (% of total)'),
    ('unemployment_pct',  'Unemployment Rate (%)'),
    ('youth_unemp_pct',   'Youth Unemployment Rate (%)'),
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
    print(f"    {'Period':<10} {'Coef':>8} {'SE':>8} {'t':>7} {'p':>7}")
    print(f"    {'-'*45}")
    for pp, col in [(-2,'pre2'),(-1,'pre1'),(0,'lag0'),(1,'lag1'),
                    (2,'lag2'),(3,'lag3'),(4,'lag4'),(5,'lag5')]:
        c_  = mod_dyn.params[col]
        se_ = mod_dyn.bse[col]
        t_  = mod_dyn.tvalues[col]
        p_  = mod_dyn.pvalues[col]
        sig_= '***' if p_<0.01 else '**' if p_<0.05 else '*' if p_<0.1 else ''
        lbl = f"Pre({abs(pp)})" if pp<0 else f"Lag({pp})"
        print(f"    {lbl:<10} {c_:>+8.3f} {se_:>8.3f} {t_:>7.2f} {p_:>7.3f} {sig_:>4}")

# ── Also run sector dose-response for employment outcomes ──
def run_sector_dose(df, outcome, label, subset_label):
    cgit = pd.read_csv('Data/GIT_Data/clean_git/cgit_clean_transactions_with_iso3.csv')
    
    for sector, sector_list in [('Transport',['Transport']),
                                  ('Energy',['Energy']),
                                  ('Metals',['Metals'])]:
        agg = (cgit[cgit['sector'].isin(sector_list)]
                   .groupby(['iso3','year'])['deal_musd'].sum()
                   .reset_index()
                   .rename(columns={'iso3':'Country Code','deal_musd':'usd'}))
        agg[f'log1p_{sector}'] = np.log1p(agg['usd'])
        df = df.merge(agg[['Country Code','year',f'log1p_{sector}']],
                     on=['Country Code','year'], how='left')
        df[f'log1p_{sector}'] = df[f'log1p_{sector}'].fillna(0)

    cols_needed = [outcome,'Country Code','year',
                   'log_gdp_pc_current_usd','population_total',
                   'percent_urban','birth_rate_crude_per_1000']
    df = df.dropna(subset=cols_needed).copy()

    print(f"\n[3] Sector Dose-Response — {subset_label} — {label}")
    print(f"    {'Sector':<15} {'Coef':>8} {'SE':>8} {'t':>7} {'p':>7}")
    print(f"    {'-'*48}")
    for sector in ['Transport','Energy','Metals']:
        var = f'log1p_{sector}'
        mod = smf.ols(
            f'{outcome} ~ {var} + {controls} + C(year) + C(Q("Country Code"))',
            data=df
        ).fit(cov_type='cluster', cov_kwds={'groups': df['Country Code']})
        c  = mod.params[var]
        se = mod.bse[var]
        t  = mod.tvalues[var]
        p  = mod.pvalues[var]
        sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        print(f"    {sector:<15} {c:>+8.4f} {se:>8.4f} {t:>7.2f} {p:>7.3f} {sig:>4}")

# ── Define subsets ──
full  = panel.copy()
lm    = panel[panel['IncomeGroup'].isin(low_mid)].copy()
ssa   = panel[panel['Region']=='Sub-Saharan Africa'].copy()

subsets = [
    (full, 'Full Panel'),
    (lm,   'Low & Middle Income'),
    (ssa,  'Sub-Saharan Africa'),
]

print("="*60)
print("  EMPLOYMENT OUTCOMES — DiD REGRESSIONS")
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

print(f"\n\n{'#'*60}")
print(f"  SECTOR DOSE-RESPONSE ON EMPLOYMENT OUTCOMES")
print(f"{'#'*60}")
for ocol, olbl in outcomes:
    for sdf, slbl in subsets:
        try:
            run_sector_dose(sdf.copy(), ocol, olbl, slbl)
        except Exception as e:
            print(f"  ERROR {slbl} {olbl}: {e}")

print("\n\nDone.")
log.close()