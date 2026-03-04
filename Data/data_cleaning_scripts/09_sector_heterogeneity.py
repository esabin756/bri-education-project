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

log = open('Data/Regressions/sector_heterogeneity_log.txt', 'w')
sys.stdout = Tee(_terminal, log)

# ── Load data ──
panel = pd.read_csv('Data/Panel/panel_with_income.csv')
cgit  = pd.read_csv('Data/GIT_Data/clean_git/cgit_clean_transactions_with_iso3.csv')

controls = 'log_gdp_pc_current_usd + population_total + percent_urban + birth_rate_crude_per_1000'

# ── Aggregate CGIT by sector to country-year ──
sectors_of_interest = {
    'Transport':   ['Transport'],
    'Energy':      ['Energy'],
    'Real Estate': ['Real Estate'],
    'Metals':      ['Metals'],
    'Extractive':  ['Metals', 'Energy'],          # combined extractive
    'Access':      ['Transport', 'Real estate'],  # combined access/construction
}

sector_aggs = {}
for label, sector_list in sectors_of_interest.items():
    agg = (cgit[cgit['sector'].isin(sector_list)]
               .groupby(['iso3','year'])
               .agg(deals=('deal_musd','count'),
                    usd=('deal_musd','sum'))
               .reset_index()
               .rename(columns={'iso3':'Country Code'}))
    agg[f'log1p_{label}'] = np.log1p(agg['usd'])
    sector_aggs[label] = agg

# ── Merge all sectors into panel ──
for label, agg in sector_aggs.items():
    panel = panel.merge(
        agg[['Country Code','year',f'log1p_{label}']],
        on=['Country Code','year'], how='left'
    )
    panel[f'log1p_{label}'] = panel[f'log1p_{label}'].fillna(0)

print("="*65)
print("  SECTOR HETEROGENEITY — DOSE RESPONSE BY SECTOR")
print("  Chinese development finance disaggregated by sector")
print("="*65)

low_mid = ['Low income','Lower middle income','Upper middle income']

outcomes = [
    ('primary_enroll_gross_pct',   'Primary Enrollment (%)'),
    ('secondary_enroll_gross_pct', 'Secondary Enrollment (%)'),
    ('tertiary_enroll_gross_pct',  'Tertiary Enrollment (%)'),
]

def run_sector(df, outcome, outcome_label, subset_label):
    cols_needed = [outcome, 'Country Code', 'year',
                   'log_gdp_pc_current_usd','population_total',
                   'percent_urban','birth_rate_crude_per_1000']
    df = df.dropna(subset=cols_needed).copy()

    print(f"\n{'='*65}")
    print(f"  {subset_label} — {outcome_label}")
    print(f"  N countries={df['Country Code'].nunique()}, N obs={len(df)}")
    print(f"{'='*65}")
    print(f"  {'Sector':<20} {'Coef':>8} {'SE':>8} {'t':>7} {'p':>7} {'':>4}")
    print(f"  {'-'*55}")

    for label in ['Transport','Energy','RealEstate','Metals','Extractive','Access']:
        var = f'log1p_{label}'
        if var not in df.columns:
            continue
        try:
            formula = (f'{outcome} ~ {var} + {controls} '
                      f'+ C(year) + C(Q("Country Code"))')
            mod = smf.ols(formula, data=df).fit(
                cov_type='cluster',
                cov_kwds={'groups': df['Country Code']}
            )
            c  = mod.params[var]
            se = mod.bse[var]
            t  = mod.tvalues[var]
            p  = mod.pvalues[var]
            sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
            print(f"  {label:<20} {c:>+8.4f} {se:>8.4f} {t:>7.2f} {p:>7.3f} {sig:>4}")
        except Exception as e:
            print(f"  {label:<20} ERROR: {e}")

# ── Run for all subsets ──
subsets = [
    (panel,                                    'Full Panel'),
    (panel[panel['IncomeGroup'].isin(low_mid)],'Low & Middle Income'),
    (panel[panel['Region']=='Sub-Saharan Africa'], 'Sub-Saharan Africa'),
]

for sdf, slbl in subsets:
    print(f"\n\n{'#'*65}")
    print(f"  {slbl}")
    print(f"{'#'*65}")
    for ocol, olbl in outcomes:
        run_sector(sdf, ocol, olbl, slbl)

# ── Joint model: all sectors together ──
print(f"\n\n{'#'*65}")
print(f"  JOINT MODEL — All sectors simultaneously (Full Panel)")
print(f"{'#'*65}")

sector_vars = ' + '.join([f'log1p_{s}' for s in ['Transport','Energy','Metals']])

for ocol, olbl in outcomes:
    cols_needed = [ocol,'Country Code','year',
                   'log_gdp_pc_current_usd','population_total',
                   'percent_urban','birth_rate_crude_per_1000']
    df = panel.dropna(subset=cols_needed).copy()

    print(f"\n  {olbl} — N={len(df)}")
    print(f"  {'Sector':<20} {'Coef':>8} {'SE':>8} {'t':>7} {'p':>7} {'':>4}")
    print(f"  {'-'*55}")

    try:
        formula = (f'{ocol} ~ {sector_vars} + {controls} '
                  f'+ C(year) + C(Q("Country Code"))')
        mod = smf.ols(formula, data=df).fit(
            cov_type='cluster',
            cov_kwds={'groups': df['Country Code']}
        )
        for s in ['Transport','Energy','Metals']:
            var = f'log1p_{s}'
            c  = mod.params[var]
            se = mod.bse[var]
            t  = mod.tvalues[var]
            p  = mod.pvalues[var]
            sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
            print(f"  {s:<20} {c:>+8.4f} {se:>8.4f} {t:>7.2f} {p:>7.3f} {sig:>4}")
    except Exception as e:
        print(f"  ERROR: {e}")

print("\n\nDone.")
log.close()