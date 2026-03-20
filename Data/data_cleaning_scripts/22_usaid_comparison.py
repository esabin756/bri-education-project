import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys, os

log = open('Data/Regressions/usaid_comparison_log.txt', 'w')
_t = sys.stdout
class Tee:
    def __init__(self,t,l): self.t=t; self.l=l
    def write(self,o): self.t.write(o); self.t.flush(); self.l.write(o); self.l.flush()
    def flush(self): self.t.flush(); self.l.flush()
sys.stdout = Tee(_t, log)

panel  = pd.read_csv('Data/Panel/panel_2023.csv')
usaid  = pd.read_csv('Data/US_Aid_Data/us_foreign_aid_usg_sector.csv')
controls = ('log_gdp_pc_current_usd + population_total + '
            'percent_urban + birth_rate_crude_per_1000')
ssa    = panel[panel['Region']=='Sub-Saharan Africa'].copy()

def stars(p):
    return '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''

# ================================================================
# PART 1: CLEAN AND AGGREGATE USAID DATA
# ================================================================
print("="*65)
print("  USAID EDUCATION AID — DATA CLEANING")
print("="*65)

# Keep only disbursements — actual money spent, not obligations or plans
# Obligations are promises to spend, disbursements are actual transfers
usaid_disb = usaid[usaid['Transaction Type Name']=='Disbursements'].copy()
print(f"\nTotal records:        {len(usaid):,}")
print(f"Disbursement records: {len(usaid_disb):,}")

# Keep only education sectors
edu_sectors = ['Basic Education', 'Higher Education',
               'Education and Social Services - General']
usaid_edu = usaid_disb[
    usaid_disb['US Sector Name'].isin(edu_sectors)
].copy()
print(f"Education records:    {len(usaid_edu):,}")

# Keep only country-level records (drop regional aggregates)
panel_codes = set(panel['Country Code'].unique())
usaid_edu = usaid_edu[
    usaid_edu['Country Code'].isin(panel_codes)
].copy()
print(f"After dropping regional: {len(usaid_edu):,}")
print(f"Countries covered:    {usaid_edu['Country Code'].nunique()}")
print(f"Years:                {usaid_edu['Fiscal Year'].min()} - "
      f"{usaid_edu['Fiscal Year'].max()}")

# Rename for merging
usaid_edu = usaid_edu.rename(columns={'Fiscal Year': 'year'})

# Aggregate to country-year total education disbursements
usaid_cy = (usaid_edu.groupby(['Country Code','year'])['constant_amount']
            .sum().reset_index()
            .rename(columns={'constant_amount': 'usaid_edu_usd'}))

# Also split by basic vs higher education
usaid_basic = (usaid_edu[usaid_edu['US Sector Name']=='Basic Education']
               .groupby(['Country Code','year'])['constant_amount']
               .sum().reset_index()
               .rename(columns={'constant_amount': 'usaid_basic_usd'}))

usaid_higher = (usaid_edu[usaid_edu['US Sector Name']=='Higher Education']
                .groupby(['Country Code','year'])['constant_amount']
                .sum().reset_index()
                .rename(columns={'constant_amount': 'usaid_higher_usd'}))

usaid_cy = usaid_cy.merge(usaid_basic,  on=['Country Code','year'], how='left')
usaid_cy = usaid_cy.merge(usaid_higher, on=['Country Code','year'], how='left')
usaid_cy = usaid_cy.fillna(0)

# Log transform investment amounts
usaid_cy['log1p_usaid_edu']    = np.log1p(usaid_cy['usaid_edu_usd']   / 1e6)
usaid_cy['log1p_usaid_basic']  = np.log1p(usaid_cy['usaid_basic_usd'] / 1e6)
usaid_cy['log1p_usaid_higher'] = np.log1p(usaid_cy['usaid_higher_usd']/ 1e6)

print(f"\nUSAID education aid summary:")
print(f"  Mean annual disbursement per country: "
      f"${usaid_cy['usaid_edu_usd'].mean()/1e6:.1f}M")
print(f"  Max annual disbursement:              "
      f"${usaid_cy['usaid_edu_usd'].max()/1e6:.1f}M")
print(f"\nTop 10 recipients (total):")
top10 = (usaid_cy.groupby('Country Code')['usaid_edu_usd']
         .sum().sort_values(ascending=False).head(10))
for cc, amt in top10.items():
    print(f"  {cc}  ${amt/1e6:.0f}M")

# ================================================================
# PART 2: BUILD USAID TREATMENT INDICATOR
# ================================================================
# Mirror the BRI treatment approach — cumulative investment threshold
# Use $10M cumulative education disbursement to match BRI edu treatment

usaid_cy_sorted = usaid_cy.sort_values(['Country Code','year'])
usaid_cy_sorted['cumul_usaid_edu'] = (usaid_cy_sorted
                                      .groupby('Country Code')['usaid_edu_usd']
                                      .cumsum())

# $10M threshold
crossed_usaid_10m = (usaid_cy_sorted[
    usaid_cy_sorted['cumul_usaid_edu'] >= 10e6]
    .groupby('Country Code')['year'].min().reset_index()
    .rename(columns={'year': 'treat_year_usaid'}))

# $50M threshold
crossed_usaid_50m = (usaid_cy_sorted[
    usaid_cy_sorted['cumul_usaid_edu'] >= 50e6]
    .groupby('Country Code')['year'].min().reset_index()
    .rename(columns={'year': 'treat_year_usaid_50m'}))

print(f"\nUSAID treatment summary:")
print(f"  $10M threshold: {len(crossed_usaid_10m)} countries treated")
print(f"  $50M threshold: {len(crossed_usaid_50m)} countries treated")

# ================================================================
# PART 3: MERGE INTO PANEL
# ================================================================
merged = panel.merge(usaid_cy, on=['Country Code','year'], how='left')
merged = merged.merge(crossed_usaid_10m, on='Country Code', how='left')
merged = merged.merge(crossed_usaid_50m, on='Country Code', how='left')

# Fill zeros for countries with no USAID education aid
for col in ['log1p_usaid_edu','log1p_usaid_basic','log1p_usaid_higher']:
    merged[col] = merged[col].fillna(0)

# Treatment indicators
merged['treat_usaid']    = merged['treat_year_usaid'].notna().astype(int)
merged['post_usaid']     = ((merged['treat_usaid']==1) &
                             (merged['year'] >= merged['treat_year_usaid'])).astype(int)
merged['treat_usaid_50'] = merged['treat_year_usaid_50m'].notna().astype(int)
merged['post_usaid_50']  = ((merged['treat_usaid_50']==1) &
                             (merged['year'] >= merged['treat_year_usaid_50m'])).astype(int)

ssa_merged = merged[merged['Region']=='Sub-Saharan Africa'].copy()

print(f"\nMerged panel: {merged.shape}")
print(f"USAID treated ($10M): {merged[merged['treat_usaid']==1]['Country Code'].nunique()}")
print(f"USAID treated ($50M): {merged[merged['treat_usaid_50']==1]['Country Code'].nunique()}")

# ================================================================
# PART 4: DOSE-RESPONSE COMPARISON — USAID vs BRI
# ================================================================
# Same specification as your BRI sector dose-response
# Compare coefficients directly

print(f"\n\n{'#'*65}")
print(f"  PART 4: DOSE-RESPONSE — USAID vs BRI EDUCATION INVESTMENT")
print(f"  Same spec: outcome ~ log_investment + controls + country FE + year FE")
print(f"  Direct comparison of coefficients")
print(f"{'#'*65}")

outcomes = [
    ('secondary_enroll_gross_pct', 'Secondary Enrollment Gross (%)'),
    ('primary_enroll_gross_pct',   'Primary Enrollment Gross (%)'),
    ('tertiary_enroll_gross_pct',  'Tertiary Enrollment Gross (%)'),
    ('secondary_net_pct',          'Secondary Enrollment Net (%)'),
]

for ocol, olbl in outcomes:
    print(f"\n  {olbl}")
    print(f"  {'Subset':<22} {'Donor':<10} {'Sector':<12} "
          f"{'Coef':>8} {'SE':>8} {'p':>7} {'':>4}")
    print(f"  {'-'*72}")

    for sdf, slbl in [(merged,'Full Panel'),(ssa_merged,'SSA')]:
        cols = [ocol,'Country Code','year',
                'log1p_usaid_edu','log1p_usaid_basic','log1p_usaid_higher',
                'log1p_Education',
                'log_gdp_pc_current_usd','population_total',
                'percent_urban','birth_rate_crude_per_1000']
        df = sdf.dropna(subset=[c for c in cols
                                 if c in sdf.columns]).copy()

        # USAID total education
        for var, donor, sector in [
            ('log1p_usaid_edu',    'USAID',  'Total Edu'),
            ('log1p_usaid_basic',  'USAID',  'Basic Edu'),
            ('log1p_usaid_higher', 'USAID',  'Higher Edu'),
            ('log1p_Education',    'China',  'Education'),
        ]:
            try:
                mod = smf.ols(
                    f'{ocol} ~ {var} + {controls} + '
                    f'C(year) + C(Q("Country Code"))',
                    data=df
                ).fit(cov_type='cluster',
                      cov_kwds={'groups': df['Country Code']})
                c  = mod.params[var]
                se = mod.bse[var]
                p  = mod.pvalues[var]
                sig = stars(p)
                print(f"  {slbl:<22} {donor:<10} {sector:<12} "
                      f"{c:>+8.4f} {se:>8.4f} {p:>7.3f} {sig:>4}")
            except Exception as e:
                print(f"  {slbl:<22} {donor:<10} {sector:<12} ERROR: {e}")

# ================================================================
# PART 5: DiD COMPARISON — USAID vs BRI TREATMENT
# ================================================================
print(f"\n\n{'#'*65}")
print(f"  PART 5: DiD COMPARISON — USAID vs BRI TREATMENT")
print(f"  Cumulative threshold treatment, same DiD spec")
print(f"{'#'*65}")

def run_did(df, outcome, treat_col, post_col):
    cols = [outcome,'Country Code','year',treat_col,post_col,
            'log_gdp_pc_current_usd','population_total',
            'percent_urban','birth_rate_crude_per_1000']
    df = df.dropna(subset=[c for c in cols
                            if c in df.columns]).copy()
    df['did'] = df[treat_col] * df[post_col]
    mod = smf.ols(
        f'{outcome} ~ did + {controls} + '
        f'C(year) + C(Q("Country Code"))',
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['Country Code']})
    return (mod.params['did'], mod.bse['did'],
            mod.pvalues['did'], int(mod.nobs),
            df['Country Code'].nunique())

for ocol, olbl in outcomes:
    print(f"\n  {olbl}")
    print(f"  {'Subset':<22} {'Donor':<10} {'Threshold':<12} "
          f"{'N Treated':>10} {'Coef':>8} {'SE':>8} {'p':>7} {'':>4}")
    print(f"  {'-'*80}")

    for sdf, slbl in [(merged,'Full Panel'),(ssa_merged,'SSA')]:
        # BRI $500M
        try:
            c,se,p,n,nc = run_did(sdf, ocol, 'treat_500m', 'post_500m')
            nt = sdf[sdf['treat_500m']==1]['Country Code'].nunique()
            print(f"  {slbl:<22} {'China':<10} {'$500M total':<12} "
                  f"{nt:>10} {c:>+8.3f} {se:>8.3f} {p:>7.3f} {stars(p):>4}")
        except Exception as e:
            print(f"  {slbl:<22} China      $500M        ERROR: {e}")

        # USAID $10M education
        try:
            c,se,p,n,nc = run_did(sdf, ocol, 'treat_usaid', 'post_usaid')
            nt = sdf[sdf['treat_usaid']==1]['Country Code'].nunique()
            print(f"  {slbl:<22} {'USAID':<10} {'$10M edu':<12} "
                  f"{nt:>10} {c:>+8.3f} {se:>8.3f} {p:>7.3f} {stars(p):>4}")
        except Exception as e:
            print(f"  {slbl:<22} USAID      $10M edu     ERROR: {e}")

        # USAID $50M education
        try:
            c,se,p,n,nc = run_did(sdf, ocol, 'treat_usaid_50', 'post_usaid_50')
            nt = sdf[sdf['treat_usaid_50']==1]['Country Code'].nunique()
            print(f"  {slbl:<22} {'USAID':<10} {'$50M edu':<12} "
                  f"{nt:>10} {c:>+8.3f} {se:>8.3f} {p:>7.3f} {stars(p):>4}")
        except Exception as e:
            print(f"  {slbl:<22} USAID      $50M edu     ERROR: {e}")

print("\n\nDone.")
log.close()