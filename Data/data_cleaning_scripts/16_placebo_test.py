import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys, os

_terminal = sys.stdout
class Tee:
    def __init__(self, t, l): self.terminal=t; self.logfile=l
    def write(self, o): self.terminal.write(o); self.terminal.flush(); self.logfile.write(o); self.logfile.flush()
    def flush(self): self.terminal.flush(); self.logfile.flush()

os.makedirs('Data/Regressions', exist_ok=True)
log = open('Data/Regressions/placebo_test_log.txt', 'w')
sys.stdout = Tee(_terminal, log)

np.random.seed(42)

panel = pd.read_csv('Data/Panel/panel_2023.csv')
controls = ('log_gdp_pc_current_usd + population_total + '
            'percent_urban + birth_rate_crude_per_1000')

def stars(p):
    return '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''

# ── Key outcomes to test ──
outcomes = [
    ('secondary_enroll_gross_pct', 'Secondary Enrollment Gross (%)'),
    ('secondary_gross_female',     'Secondary Gross Female (%)'),
    ('secondary_gross_male',       'Secondary Gross Male (%)'),
    ('tertiary_enroll_gross_pct',  'Tertiary Enrollment Gross (%)'),
    ('gdp_growth',                 'GDP Growth (%)'),
    ('female_emp_ratio',           'Female Employment Ratio (%)'),
]

# ── Subsets ──
low_mid = ['Low income','Lower middle income','Upper middle income']
ssa     = panel[panel['Region']=='Sub-Saharan Africa'].copy()

print("="*65)
print("  PLACEBO TEST")
print("  Randomly assign fake treatment years to never-treated")
print("  countries and re-run main DiD specification")
print("  N iterations: 500")
print("="*65)

def run_placebo_iter(df, outcome, seed):
    """Single placebo iteration — assign random treatment years."""
    np.random.seed(seed)
    df = df.copy()

    # Get never-treated countries
    never = df[df['treat_500m']==0]['Country Code'].unique()
    if len(never) < 10:
        return None

    # Get distribution of actual treatment years to draw from
    treat_years = panel[panel['treat_500m']==1]['treat_year'].dropna().unique()

    # Assign random treatment year to each never-treated country
    fake_treat = pd.DataFrame({
        'Country Code': never,
        'fake_treat_year': np.random.choice(treat_years, size=len(never))
    })

    df = df[df['treat_500m']==0].merge(fake_treat, on='Country Code', how='left')
    df['treat_placebo'] = 1
    df['post_placebo']  = (df['year'] >= df['fake_treat_year']).astype(int)
    df['did_placebo']   = df['treat_placebo'] * df['post_placebo']

    cols = [outcome,'Country Code','year','did_placebo',
            'log_gdp_pc_current_usd','population_total',
            'percent_urban','birth_rate_crude_per_1000']
    df = df.dropna(subset=[c for c in cols if c in df.columns]).copy()
    if len(df) < 100:
        return None

    try:
        mod = smf.ols(
            f'Q("{outcome}") ~ did_placebo + {controls} + '
            f'C(year) + C(Q("Country Code"))',
            data=df
        ).fit(cov_type='cluster', cov_kwds={'groups': df['Country Code']})
        return mod.params['did_placebo'], mod.pvalues['did_placebo']
    except:
        return None

# ================================================================
# PART 1: DISTRIBUTION OF PLACEBO COEFFICIENTS
# ================================================================
print("\n\n" + "#"*65)
print("  PART 1: PLACEBO COEFFICIENT DISTRIBUTION (500 iterations)")
print("#"*65)

N_ITER = 500

for ocol, olbl in outcomes:
    for sdf, slbl in [(panel, 'Full Panel'), (ssa, 'Sub-Saharan Africa')]:
        print(f"\n  {olbl} — {slbl}")

        coefs = []
        pvals = []
        sig_count = 0

        for i in range(N_ITER):
            result = run_placebo_iter(sdf, ocol, seed=i)
            if result is not None:
                c, p = result
                coefs.append(c)
                pvals.append(p)
                if p < 0.05:
                    sig_count += 1

        if len(coefs) == 0:
            print("  No valid iterations")
            continue

        coefs = np.array(coefs)
        pvals = np.array(pvals)

        print(f"  Valid iterations:     {len(coefs)}")
        print(f"  Mean placebo coef:    {np.mean(coefs):+.3f}")
        print(f"  Std placebo coef:     {np.std(coefs):.3f}")
        print(f"  95th pctile |coef|:   {np.percentile(np.abs(coefs), 95):.3f}")
        print(f"  % significant (p<.05): {sig_count/len(coefs)*100:.1f}%")
        print(f"  % positive coef:      {(coefs>0).mean()*100:.1f}%")

# ================================================================
# PART 2: COMPARE ACTUAL VS PLACEBO
# ================================================================
print("\n\n" + "#"*65)
print("  PART 2: ACTUAL ESTIMATES VS PLACEBO DISTRIBUTION")
print("#"*65)

def run_actual(df, outcome):
    cols = [outcome,'Country Code','year','treat_500m','post_500m',
            'log_gdp_pc_current_usd','population_total',
            'percent_urban','birth_rate_crude_per_1000']
    df = df.dropna(subset=[c for c in cols if c in df.columns]).copy()
    df['did'] = df['treat_500m'] * df['post_500m']
    mod = smf.ols(
        f'{outcome} ~ did + {controls} + C(year) + C(Q("Country Code"))',
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['Country Code']})
    return mod.params['did'], mod.pvalues['did']

print(f"\n  {'Outcome':<35} {'Subset':<25} {'Actual':>8} {'p':>7} "
      f"{'Placebo mean':>13} {'Placebo SD':>11} {'Rank pct':>9}")
print(f"  {'-'*110}")

for ocol, olbl in outcomes:
    for sdf, slbl in [(panel, 'Full Panel'), (ssa, 'Sub-Saharan Africa')]:
        try:
            actual_c, actual_p = run_actual(sdf, ocol)
        except:
            continue

        # Re-run placebo to get distribution
        coefs = []
        for i in range(N_ITER):
            result = run_placebo_iter(sdf, ocol, seed=i)
            if result is not None:
                coefs.append(result[0])

        if len(coefs) == 0:
            continue

        coefs = np.array(coefs)
        placebo_mean = np.mean(coefs)
        placebo_sd   = np.std(coefs)

        # Where does actual estimate rank in placebo distribution?
        rank_pct = (coefs < actual_c).mean() * 100
        sig = stars(actual_p)

        print(f"  {olbl:<35} {slbl:<25} {actual_c:>+8.3f} "
              f"{actual_p:>7.3f} {placebo_mean:>+13.3f} "
              f"{placebo_sd:>11.3f} {rank_pct:>8.1f}%")

# ================================================================
# PART 3: PLACEBO ON EDUCATION TREATMENT
# ================================================================
print("\n\n" + "#"*65)
print("  PART 3: PLACEBO — EDUCATION-SPECIFIC TREATMENT ($50M)")
print("#"*65)

def run_placebo_edu_iter(df, outcome, seed):
    np.random.seed(seed)
    df = df.copy()

    never = df[df['treat_edu']==0]['Country Code'].unique()
    if len(never) < 10:
        return None

    treat_years = panel[panel['treat_edu']==1]['treat_year_edu'].dropna().unique()
    fake_treat  = pd.DataFrame({
        'Country Code':    never,
        'fake_treat_year': np.random.choice(treat_years, size=len(never))
    })

    df = df[df['treat_edu']==0].merge(fake_treat, on='Country Code', how='left')
    df['post_placebo'] = (df['year'] >= df['fake_treat_year']).astype(int)
    df['did_placebo']  = df['post_placebo'].copy()

    cols = [outcome,'Country Code','year','did_placebo',
            'log_gdp_pc_current_usd','population_total',
            'percent_urban','birth_rate_crude_per_1000']
    df = df.dropna(subset=[c for c in cols if c in df.columns]).copy()
    if len(df) < 100:
        return None

    try:
        mod = smf.ols(
            f'{outcome} ~ did_placebo + {controls} + '
            f'C(year) + C(Q("Country Code"))',
            data=df
        ).fit(cov_type='cluster', cov_kwds={'groups': df['Country Code']})
        return mod.params['did_placebo'], mod.pvalues['did_placebo']
    except:
        return None

edu_outcomes = [
    ('secondary_enroll_gross_pct', 'Secondary Enrollment Gross'),
    ('tertiary_enroll_gross_pct',  'Tertiary Enrollment Gross'),
]

for ocol, olbl in edu_outcomes:
    for sdf, slbl in [(panel, 'Full Panel'), (ssa, 'Sub-Saharan Africa')]:
        print(f"\n  {olbl} — {slbl}")
        coefs = []
        for i in range(N_ITER):
            result = run_placebo_edu_iter(sdf, ocol, seed=i)
            if result is not None:
                coefs.append(result[0])

        if not coefs:
            continue

        coefs = np.array(coefs)
        sig_count = sum(1 for i in range(N_ITER)
                       if run_placebo_edu_iter(sdf, ocol, seed=i) is not None
                       and run_placebo_edu_iter(sdf, ocol, seed=i)[1] < 0.05)

        print(f"  Mean placebo coef:     {np.mean(coefs):+.3f}")
        print(f"  Std placebo coef:      {np.std(coefs):.3f}")
        print(f"  % significant (p<.05): {sig_count/len(coefs)*100:.1f}%")

print("\n\nDone.")
log.close()