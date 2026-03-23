import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys, os

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

os.makedirs('Data/Regressions', exist_ok=True)
log = open('Data/Regressions/robustness_thresholds_log.txt', 'w')
sys.stdout = Tee(_terminal, log)

# ── Load current panel (205 real countries, aggregates already dropped) ──
panel_base = pd.read_csv('Data/Panel/panel_2023.csv')

# ── Load current AidData ──
aid = pd.read_csv('Data/clean_aid_data/aiddata_clg_country_year.csv')

print("="*70)
print("  ROBUSTNESS CHECK — ALTERNATIVE INVESTMENT THRESHOLDS")
print(f"  Panel: {panel_base['Country Code'].nunique()} countries, "
      f"{panel_base['year'].min()}-{panel_base['year'].max()}")
print("  Main spec: $500M | Alternatives: $250M, $750M, $1B, $2B")
print("="*70)

controls = ('log_gdp_pc_current_usd + population_total + '
            'percent_urban + birth_rate_crude_per_1000')

outcomes = [
    ('primary_enroll_gross_pct',   'Primary Enrollment (%)'),
    ('secondary_enroll_gross_pct', 'Secondary Enrollment (%)'),
    ('tertiary_enroll_gross_pct',  'Tertiary Enrollment (%)'),
]

def stars(p):
    return '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''

def build_treatment(aid, threshold_m):
    """Build treat_year for given cumulative threshold in USD millions."""
    aid_sorted = aid.sort_values(['Country Code','year'])
    aid_sorted['cumul_usd'] = (aid_sorted
                               .groupby('Country Code')['usd_const_2023']
                               .cumsum())
    crossed = (aid_sorted[aid_sorted['cumul_usd'] >= threshold_m * 1e6]
               .groupby('Country Code')['year']
               .min().reset_index()
               .rename(columns={'year': 'treat_year'}))
    return crossed

def run_did(panel, outcome):
    cols = [outcome,'Country Code','year',
            'treat_500m','post_500m',
            'log_gdp_pc_current_usd','population_total',
            'percent_urban','birth_rate_crude_per_1000']
    df = panel.dropna(subset=[c for c in cols
                               if c in panel.columns]).copy()
    df['did'] = df['treat_500m'] * df['post_500m']
    mod = smf.ols(
        f'Q("{outcome}") ~ did + {controls} + '
        f'C(year) + C(Q("Country Code"))',
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['Country Code']})
    return (mod.params['did'], mod.bse['did'],
            mod.tvalues['did'], mod.pvalues['did'],
            int(mod.nobs), df['Country Code'].nunique())

thresholds = [250, 500, 750, 1000, 2000]

for ocol, olbl in outcomes:
    print(f"\n\n{'#'*70}")
    print(f"  OUTCOME: {olbl}")
    print(f"{'#'*70}")
    print(f"\n  {'Threshold':>12} {'N Treated':>10} {'N Obs':>8} "
          f"{'Coef':>10} {'SE':>8} {'t':>7} {'p':>7} {'':>4}")
    print(f"  {'-'*70}")

    for thresh in thresholds:
        try:
            crossed = build_treatment(aid, thresh)

            # Rebuild treatment on current panel
            panel = panel_base.copy()
            panel = panel.drop(
                columns=[c for c in ['treat_500m','post_500m','treat_year']
                         if c in panel.columns], errors='ignore')
            panel = panel.merge(
                crossed[['Country Code','treat_year']],
                on='Country Code', how='left')
            panel['treat_500m'] = panel['treat_year'].notna().astype(int)
            panel['post_500m']  = (
                (panel['treat_500m']==1) &
                (panel['year'] >= panel['treat_year'])
            ).astype(int)

            n_treated = panel[panel['treat_500m']==1]['Country Code'].nunique()
            c, se, t, p, n, nc = run_did(panel, ocol)
            sig    = stars(p)
            marker = ' <- MAIN SPEC' if thresh == 500 else ''
            print(f"  ${thresh:>9}M {n_treated:>10} {n:>8} "
                  f"{c:>+10.3f} {se:>8.3f} {t:>7.2f} "
                  f"{p:>7.3f} {sig:>4}{marker}")
        except Exception as e:
            print(f"  ${thresh:>9}M  ERROR: {e}")

# ── SSA robustness ──
print(f"\n\n{'#'*70}")
print(f"  ROBUSTNESS — SUB-SAHARAN AFRICA ONLY")
print(f"{'#'*70}")

for ocol, olbl in outcomes:
    print(f"\n  {olbl}")
    print(f"  {'Threshold':>12} {'N Treated':>10} {'N Obs':>8} "
          f"{'Coef':>10} {'SE':>8} {'t':>7} {'p':>7} {'':>4}")
    print(f"  {'-'*65}")

    for thresh in thresholds:
        try:
            crossed = build_treatment(aid, thresh)

            panel = panel_base.copy()
            panel = panel.drop(
                columns=[c for c in ['treat_500m','post_500m','treat_year']
                         if c in panel.columns], errors='ignore')
            panel = panel.merge(
                crossed[['Country Code','treat_year']],
                on='Country Code', how='left')
            panel['treat_500m'] = panel['treat_year'].notna().astype(int)
            panel['post_500m']  = (
                (panel['treat_500m']==1) &
                (panel['year'] >= panel['treat_year'])
            ).astype(int)

            ssa = panel[panel['Region']=='Sub-Saharan Africa'].copy()
            n_treated = ssa[ssa['treat_500m']==1]['Country Code'].nunique()
            c, se, t, p, n, nc = run_did(ssa, ocol)
            sig    = stars(p)
            marker = ' <- MAIN SPEC' if thresh == 500 else ''
            print(f"  ${thresh:>9}M {n_treated:>10} {n:>8} "
                  f"{c:>+10.3f} {se:>8.3f} {t:>7.2f} "
                  f"{p:>7.3f} {sig:>4}{marker}")
        except Exception as e:
            print(f"  ${thresh:>9}M  ERROR: {e}")

print("\n\nDone.")
log.close()