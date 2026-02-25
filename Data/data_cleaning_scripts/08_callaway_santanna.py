import pyfixest as pf
from pyfixest.estimation.estimation import feols
import pandas as pd
import numpy as np
import pyfixest as pf

panel = pd.read_csv('Data/Panel/panel_with_income.csv')
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

log = open('Data/Regressions/cs_did_log.txt', 'w')
sys.stdout = Tee(_terminal, log)

# ── CS-DiD requires: never-treated coded as 0 in treat_year ──
panel['g'] = panel['treat_year'].fillna(0).astype(int)

# ── Controls ──
controls = 'log_gdp_pc_current_usd + population_total + percent_urban + birth_rate_crude_per_1000'

low_mid = ['Low income','Lower middle income','Upper middle income']

def run_cs(df, outcome, label, subset_label):
    df = df.copy().dropna(subset=[outcome, 'log_gdp_pc_current_usd',
                                   'population_total', 'percent_urban',
                                   'birth_rate_crude_per_1000'])
    df = df[df[outcome].notna()].copy()

    print(f"\n{'='*60}")
    print(f"  {subset_label} — {label}")
    print(f"  N countries={df['Country Code'].nunique()}, N obs={len(df)}")
    print(f"{'='*60}")

    try:
        cs = pf.feols(
            f'{outcome} ~ sunab(g, year) + {controls} | Country Code + year',
            data=df,
            vcov={'CRV1': 'Country Code'}
        )
        print(cs.summary())

        # ATT — average treatment effect on treated
        print("\n  Sun & Abraham ATT decomposition printed above.")
        print("  Look for 'ATT' aggregate and event-time coefficients.")

    except Exception as e:
        print(f"  ERROR: {e}")

# ── Define subsets ──
full    = panel.copy()
lm_only = panel[panel['IncomeGroup'].isin(low_mid)].copy()
ssa     = panel[panel['Region']=='Sub-Saharan Africa'].copy()

outcomes = [
    ('primary_enroll_gross_pct',   'Primary Enrollment (%)'),
    ('secondary_enroll_gross_pct', 'Secondary Enrollment (%)'),
    ('tertiary_enroll_gross_pct',  'Tertiary Enrollment (%)'),
]

print("="*60)
print("  CALLAWAY-SANTANNA / SUN-ABRAHAM DiD")
print("  Heterogeneity-robust staggered treatment estimator")
print("="*60)

for sdf, slbl in [
    (full,    'Full Panel'),
    (lm_only, 'Low & Middle Income'),
    (ssa,     'Sub-Saharan Africa'),
]:
    for col, lbl in outcomes:
        run_cs(sdf, col, lbl, slbl)

print("\n\nDone.")
log.close()