import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys, os

log = open('Data/Regressions/usaid_comparison_v2_log.txt', 'w')
_t = sys.stdout
class Tee:
    def __init__(self, t, l):
        self.t = t
        self.l = l
    def write(self, o):
        # write to both the original stdout and the log file
        self.t.write(o)
        self.t.flush()
        self.l.write(o)
        self.l.flush()
    def flush(self):
        self.t.flush()
        self.l.flush()

# replace stdout with a tee object
sys.stdout = Tee(_t, log)

panel  = pd.read_csv('Data/Panel/panel_2023.csv')
usaid  = pd.read_csv('Data/US_Aid_Data/us_foreign_aid_usg_sector.csv')
controls = ('log_gdp_pc_current_usd + population_total + '
            'percent_urban + birth_rate_crude_per_1000')

def stars(p):
    return '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''

# ================================================================
# STEP 1: CLEAN USAID — same as before but drop WLD
# ================================================================
print("="*65)
print("  USAID vs CHINA EDUCATION AID — RESTRICTED SAMPLE COMPARISON")
print("  Strategy: compare dose-response coefficients within the")
print("  same set of countries and years that received BOTH types")
print("  of aid. Controls for targeting differences.")
print("="*65)

panel_codes = set(panel['Country Code'].unique())

usaid_edu = usaid[
    (usaid['Transaction Type Name'] == 'Disbursements') &
    (usaid['US Sector Name'].isin(['Basic Education',
                                   'Higher Education',
                                   'Education and Social Services - General'])) &
    (usaid['Country Code'].isin(panel_codes)) &
    (usaid['Country Code'] != 'WLD')   # drop world aggregate
].copy()

usaid_edu = usaid_edu.rename(columns={'Fiscal Year': 'year'})

# Restrict to 2001-2023 to match panel
usaid_edu = usaid_edu[usaid_edu['year'].between(2001, 2023)]

usaid_cy = (usaid_edu.groupby(['Country Code','year'])['constant_amount']
            .sum().reset_index()
            .rename(columns={'constant_amount': 'usaid_edu_usd'}))

# Also get basic education separately
usaid_basic = (usaid_edu[usaid_edu['US Sector Name']=='Basic Education']
               .groupby(['Country Code','year'])['constant_amount']
               .sum().reset_index()
               .rename(columns={'constant_amount': 'usaid_basic_usd'}))

usaid_cy = usaid_cy.merge(usaid_basic, on=['Country Code','year'], how='left')
usaid_cy['usaid_basic_usd'] = usaid_cy['usaid_basic_usd'].fillna(0)

# Log transform
usaid_cy['log1p_usaid_edu']   = np.log1p(usaid_cy['usaid_edu_usd']   / 1e6)
usaid_cy['log1p_usaid_basic'] = np.log1p(usaid_cy['usaid_basic_usd'] / 1e6)

# ================================================================
# STEP 2: MERGE AND BUILD RESTRICTED SAMPLE
# ================================================================
merged = panel.merge(usaid_cy, on=['Country Code','year'], how='left')
merged['log1p_usaid_edu']   = merged['log1p_usaid_edu'].fillna(0)
merged['log1p_usaid_basic'] = merged['log1p_usaid_basic'].fillna(0)

# Restricted sample: countries that received BOTH
# Chinese investment (any) AND USAID education aid (any)
# This makes the comparison apples-to-apples on targeting
received_china = set(
    panel[panel['log1p_usd_const_2023'] > 0]['Country Code'].unique()
)
received_usaid = set(
    usaid_cy[usaid_cy['usaid_edu_usd'] > 0]['Country Code'].unique()
)
both = received_china & received_usaid

print(f"\nCountries receiving Chinese investment:    {len(received_china)}")
print(f"Countries receiving USAID education aid:   {len(received_usaid)}")
print(f"Countries receiving BOTH:                  {len(both)}")

restricted = merged[merged['Country Code'].isin(both)].copy()
ssa_restricted = restricted[
    restricted['Region'] == 'Sub-Saharan Africa'
].copy()

print(f"\nRestricted panel: {restricted.shape}")
print(f"SSA restricted:   {ssa_restricted.shape}")
print(f"SSA countries:    {ssa_restricted['Country Code'].nunique()}")

# ================================================================
# STEP 3: DOSE-RESPONSE COMPARISON WITHIN RESTRICTED SAMPLE
# ================================================================
# Run the SAME regression for both donors:
#   outcome ~ log_investment + controls + country FE + year FE
#
# By using the same sample and same spec, any difference in
# coefficients reflects donor effectiveness not targeting

print(f"\n\n{'#'*65}")
print(f"  DOSE-RESPONSE COMPARISON — RESTRICTED SAMPLE")
print(f"  Only countries receiving BOTH Chinese and USAID education aid")
print(f"  Same regression spec for both donors — fair comparison")
print(f"{'#'*65}")

outcomes = [
    ('secondary_enroll_gross_pct', 'Secondary Enrollment Gross (%)'),
    ('primary_enroll_gross_pct',   'Primary Enrollment Gross (%)'),
    ('tertiary_enroll_gross_pct',  'Tertiary Enrollment Gross (%)'),
    ('secondary_net_pct',          'Secondary Enrollment Net (%)'),
    ('gdp_growth',                 'GDP Growth (%)'),
]

invest_vars = [
    ('log1p_usaid_edu',   'USAID', 'Total Education'),
    ('log1p_usaid_basic', 'USAID', 'Basic Education'),
    ('log1p_Education',   'China', 'Education'),
    ('log1p_Transport',   'China', 'Transport'),
    ('log1p_Energy',      'China', 'Energy'),
    ('log1p_Health',      'China', 'Health'),
]

for ocol, olbl in outcomes:
    print(f"\n  {olbl}")
    for sdf, slbl in [
        (restricted,     'Full Restricted Sample'),
        (ssa_restricted, 'SSA Restricted'),
    ]:
        n_countries = sdf['Country Code'].nunique()
        print(f"\n    {slbl} ({n_countries} countries)")
        print(f"    {'Donor':<8} {'Sector':<18} {'Coef':>8} "
              f"{'SE':>8} {'p':>7} {'N':>7} {'':>4}")
        print(f"    {'-'*58}")

        cols_needed = [ocol,'Country Code','year',
                       'log_gdp_pc_current_usd','population_total',
                       'percent_urban','birth_rate_crude_per_1000']
        df = sdf.dropna(subset=cols_needed).copy()

        for var, donor, sector in invest_vars:
            if var not in df.columns:
                continue
            try:
                mod = smf.ols(
                    f'{ocol} ~ {var} + {controls} + '
                    f'C(year) + C(Q("Country Code"))',
                    data=df
                ).fit(cov_type='cluster',
                      cov_kwds={'groups': df['Country Code']})
                c   = mod.params[var]
                se  = mod.bse[var]
                p   = mod.pvalues[var]
                sig = stars(p)
                # Flag significant results clearly
                flag = ' <--' if p < 0.1 else ''
                print(f"    {donor:<8} {sector:<18} {c:>+8.4f} "
                      f"{se:>8.4f} {p:>7.3f} {int(mod.nobs):>7} "
                      f"{sig:>4}{flag}")
            except Exception as e:
                print(f"    {donor:<8} {sector:<18} ERROR: {e}")

# ================================================================
# STEP 4: SUMMARY TABLE FOR THESIS
# ================================================================
print(f"\n\n{'#'*65}")
print(f"  SUMMARY: KEY COEFFICIENTS FOR THESIS TABLE")
print(f"  Secondary Enrollment — Restricted Sample Comparison")
print(f"{'#'*65}")

print(f"\n  {'Sample':<25} {'Donor':<8} {'Sector':<18} "
      f"{'Coef':>8} {'p':>7} {'Interpretation'}")
print(f"  {'-'*80}")

key_specs = [
    ('log1p_usaid_edu',   'USAID', 'Total Education'),
    ('log1p_usaid_basic', 'USAID', 'Basic Education'),
    ('log1p_Education',   'China', 'Education'),
    ('log1p_Transport',   'China', 'Transport'),
    ('log1p_Energy',      'China', 'Energy'),
]

for sdf, slbl in [
    (restricted,     'Full Restricted'),
    (ssa_restricted, 'SSA Restricted'),
]:
    cols_needed = ['secondary_enroll_gross_pct','Country Code','year',
                   'log_gdp_pc_current_usd','population_total',
                   'percent_urban','birth_rate_crude_per_1000']
    df = sdf.dropna(subset=cols_needed).copy()
    for var, donor, sector in key_specs:
        if var not in df.columns:
            continue
        try:
            mod = smf.ols(
                f'secondary_enroll_gross_pct ~ {var} + {controls} + '
                f'C(year) + C(Q("Country Code"))',
                data=df
            ).fit(cov_type='cluster',
                  cov_kwds={'groups': df['Country Code']})
            c   = mod.params[var]
            p   = mod.pvalues[var]
            sig = stars(p)
            interp = ('Positive*' if p<0.1 and c>0
                      else 'Negative*' if p<0.1 and c<0
                      else 'Null')
            print(f"  {slbl:<25} {donor:<8} {sector:<18} "
                  f"{c:>+8.4f} {p:>7.3f} {sig:>4}  {interp}")
        except:
            pass

print("\n\nDone.")

# restore the original stdout before closing the log so that the
# Tee.flush method doesn't try to write to a closed file
sys.stdout = _t
log.close()