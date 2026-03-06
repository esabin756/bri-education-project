import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys, os

# ── Logging setup: prints to both terminal and file simultaneously ──
log = open('Data/Regressions/action_plan_robustness_log.txt', 'w')
_t = sys.stdout
class Tee:
    def __init__(self,t,l): self.t=t; self.l=l
    def write(self,o): self.t.write(o); self.t.flush(); self.l.write(o); self.l.flush()
    def flush(self): self.t.flush(); self.l.flush()
sys.stdout = Tee(_t, log)

# ── Load the main panel ──
panel = pd.read_csv('Data/Panel/panel_2023.csv')

# ── Control variables included in every regression ──
# These absorb variation in GDP, population, urbanization, and birth rates
# so the treatment effect isn't picking up country-level development trends
controls = ('log_gdp_pc_current_usd + population_total + '
            'percent_urban + birth_rate_crude_per_1000')

# ── SSA subset ──
ssa = panel[panel['Region']=='Sub-Saharan Africa'].copy()

def stars(p):
    return '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''

# ── Outcomes to test ──
# We focus on the three most relevant enrollment outcomes
outcomes = [
    ('primary_enroll_gross_pct',   'Primary Enrollment Gross (%)'),  # add this
    ('secondary_enroll_gross_pct', 'Secondary Enrollment Gross (%)'),
    ('tertiary_enroll_gross_pct',  'Tertiary Enrollment Gross (%)'),
    ('secondary_net_pct',          'Secondary Enrollment Net (%)'),
]

# ── Cutoff years to test ──
# Our main spec uses 2016 (when China announced the Education Action Plan)
# We test 2014-2018 to check if 2016 is special or just part of a general trend
# If results only appear at 2016 and not nearby years, that supports our story
# If results appear at ALL years, it suggests a general trend not specific to the plan
cutoff_years = [2014, 2015, 2016, 2017, 2018]

print("="*65)
print("  ACTION PLAN ROBUSTNESS — ALTERNATIVE CUTOFF YEARS")
print("  Testing sensitivity of 2016 cutoff assumption")
print("  Key question: is 2016 special, or just part of a trend?")
print("="*65)

# ================================================================
# PART 1: ALTERNATIVE CUTOFF YEARS
# ================================================================
# For each outcome and subset, we run the same interaction regression
# but change the cutoff year from 2016 to nearby years.
#
# The regression is:
#   outcome = β1(Treated x Post-Cutoff) + β2(Treated) + β3(Post-Cutoff)
#             + controls + country FE + year FE
#
# β1 is our coefficient of interest — it captures whether treated countries
# diverged from control countries specifically after the cutoff year.
#
# Country FE: absorb all time-invariant country characteristics
# Year FE: absorb global trends affecting all countries equally
# Clustering: standard errors clustered by country for serial correlation

for ocol, olbl in outcomes:
    print(f"\n{'#'*65}")
    print(f"  {olbl}")
    print(f"{'#'*65}")

    for sdf, slbl in [(panel,'Full Panel'),(ssa,'Sub-Saharan Africa')]:
        print(f"\n  {slbl}")
        print(f"  {'Cutoff':<10} {'Coef':>8} {'SE':>8} "
              f"{'p':>7} {'N':>7} {'':>4}")
        print(f"  {'-'*45}")

        for yr in cutoff_years:
            df = sdf.copy()

            # Create post-cutoff dummy: 1 if year >= cutoff, 0 otherwise
            df['post_cut'] = (df['year'] >= yr).astype(int)

            # Interaction term: Treated x Post-Cutoff
            # This is our DiD coefficient — it asks:
            # "Did treated countries change differently after the cutoff
            #  compared to never-treated countries?"
            df['treat_x_post'] = df['treat_500m'] * df['post_cut']

            cols = [ocol,'Country Code','year','treat_500m',
                    'post_cut','treat_x_post',
                    'log_gdp_pc_current_usd','population_total',
                    'percent_urban','birth_rate_crude_per_1000']
            df = df.dropna(subset=[c for c in cols
                                    if c in df.columns]).copy()
            try:
                mod = smf.ols(
                    f'{ocol} ~ treat_x_post + treat_500m + '
                    f'post_cut + {controls} + '
                    f'C(year) + C(Q("Country Code"))',
                    data=df
                ).fit(cov_type='cluster',
                      cov_kwds={'groups': df['Country Code']})

                c   = mod.params['treat_x_post']
                se  = mod.bse['treat_x_post']
                p   = mod.pvalues['treat_x_post']
                sig = stars(p)

                # Flag our main specification
                marker = ' <- MAIN SPEC' if yr == 2016 else ''
                print(f"  {yr:<10} {c:>+8.3f} {se:>8.3f} "
                      f"{p:>7.3f} {int(mod.nobs):>7} {sig:>4}{marker}")

            except Exception as e:
                print(f"  {yr:<10} ERROR: {e}")

# ================================================================
# PART 2: PLACEBO CUTOFF ON NEVER-TREATED COUNTRIES
# ================================================================
# This is the key validity check. We take only the never-treated countries
# (those that never received $500M+ in BRI investment) and assign them
# a FAKE 2016 cutoff.
#
# Logic: if the 2016 result is real and driven by BRI investment,
# then applying the same 2016 cutoff to countries that never received
# BRI investment should produce NO significant result.
#
# If the placebo IS significant, it means something else happened in 2016
# globally that is driving our result — not the Education Action Plan.
# This would be a problem for our interpretation.

print(f"\n\n{'#'*65}")
print(f"  PART 2: PLACEBO — FAKE 2016 CUTOFF ON NEVER-TREATED COUNTRIES")
print(f"  If our finding is real: placebo should be null")
print(f"  If placebo is significant: 2016 result is just a global trend")
print(f"{'#'*65}")

# Never-treated: countries that never crossed the $500M BRI threshold
never     = panel[panel['treat_500m']==0].copy()
never_ssa = never[never['Region']=='Sub-Saharan Africa'].copy()

for sdf, slbl in [(never,     'Full Panel Never-Treated'),
                  (never_ssa, 'SSA Never-Treated')]:

    print(f"\n  {slbl} (N countries={sdf['Country Code'].nunique()})")
    print(f"  {'Outcome':<35} {'Coef':>8} {'SE':>8} "
          f"{'p':>7} {'':>4}")
    print(f"  {'-'*58}")

    for ocol, olbl in outcomes:
        df = sdf.copy()

        # Apply the same 2016 cutoff to never-treated countries
        # Since they never received BRI investment, any effect we find
        # here is spurious — it's picking up something else that happened
        # in 2016 globally
        df['post_2016']        = (df['year'] >= 2016).astype(int)
        df['treat_x_post2016'] = df['post_2016'].copy()  # all "treated" here

        cols = [ocol,'Country Code','year','post_2016',
                'treat_x_post2016',
                'log_gdp_pc_current_usd','population_total',
                'percent_urban','birth_rate_crude_per_1000']
        df = df.dropna(subset=[c for c in cols
                                if c in df.columns]).copy()
        try:
            mod = smf.ols(
                f'{ocol} ~ treat_x_post2016 + post_2016 + '
                f'{controls} + C(year) + C(Q("Country Code"))',
                data=df
            ).fit(cov_type='cluster',
                  cov_kwds={'groups': df['Country Code']})

            c   = mod.params['treat_x_post2016']
            se  = mod.bse['treat_x_post2016']
            p   = mod.pvalues['treat_x_post2016']
            sig = stars(p)
            print(f"  {olbl:<35} {c:>+8.3f} {se:>8.3f} "
                  f"{p:>7.3f} {sig:>4}")

        except Exception as e:
            print(f"  {olbl:<35} ERROR: {e}")

print("\n\nDone.")
log.close()