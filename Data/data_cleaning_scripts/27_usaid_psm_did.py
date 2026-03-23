import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats as scipy_stats
import sys, os

os.makedirs('Data/Regressions', exist_ok=True)
results_path = 'Data/Regressions/27_usaid_psm_did_results.txt'
log = open(results_path, 'w')
_t = sys.stdout
class Tee:
    def __init__(self,t,l): self.t=t; self.l=l
    def write(self,o): self.t.write(o); self.t.flush(); self.l.write(o); self.l.flush()
    def flush(self): self.t.flush(); self.l.flush()
sys.stdout = Tee(_t, log)

panel = pd.read_csv('Data/Panel/panel_2023.csv')
usaid = pd.read_csv('Data/US_Aid_Data/us_foreign_aid_usg_sector.csv')

controls = ('log_gdp_pc_current_usd + population_total + '
            'percent_urban + birth_rate_crude_per_1000')

def stars(p):
    return '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''

print("="*65)
print("  USAID vs CHINA — PSM DiD COMPARISON")
print(f"  Panel: {panel['Country Code'].nunique()} countries")
print(f"  Writing results to: {results_path}")
print("="*65)

# ── Clean USAID ──
panel_codes = set(panel['Country Code'].unique())
usaid_edu = usaid[
    (usaid['Transaction Type Name'] == 'Disbursements') &
    (usaid['US Sector Name'].isin(['Basic Education',
                                   'Higher Education',
                                   'Education and Social Services - General'])) &
    (usaid['Country Code'].isin(panel_codes)) &
    (usaid['Country Code'] != 'WLD')
].copy()
usaid_edu = usaid_edu.rename(columns={'Fiscal Year': 'year'})
usaid_edu = usaid_edu[usaid_edu['year'].between(2000, 2023)]

usaid_cy = (usaid_edu.groupby(['Country Code','year'])['constant_amount']
            .sum().reset_index()
            .rename(columns={'constant_amount': 'usaid_edu_usd'}))

# ── Build USAID treatment ($10M cumulative) ──
usaid_sorted = usaid_cy.sort_values(['Country Code','year'])
usaid_sorted['cumul_usaid'] = (usaid_sorted
                               .groupby('Country Code')['usaid_edu_usd']
                               .cumsum())
crossed_usaid = (usaid_sorted[usaid_sorted['cumul_usaid'] >= 10e6]
                 .groupby('Country Code')['year'].min()
                 .reset_index()
                 .rename(columns={'year': 'treat_year_usaid'}))

panel = panel.merge(usaid_cy, on=['Country Code','year'], how='left')
panel['usaid_edu_usd']   = panel['usaid_edu_usd'].fillna(0)
panel['log1p_usaid_edu'] = np.log1p(panel['usaid_edu_usd'] / 1e6)
panel = panel.drop(columns=[c for c in panel.columns
                              if 'treat_year_usaid' in c], errors='ignore')
panel = panel.merge(crossed_usaid, on='Country Code', how='left')
panel['treat_usaid'] = panel['treat_year_usaid'].notna().astype(int)
panel['post_usaid']  = ((panel['treat_usaid']==1) &
                         (panel['year'] >= panel['treat_year_usaid'])).astype(int)

print(f"\nUSAID treated ($10M+): "
      f"{panel[panel['treat_usaid']==1]['Country Code'].nunique()}")
print(f"Never treated:         "
      f"{panel[panel['treat_usaid']==0]['Country Code'].nunique()}")

# ================================================================
# PSM: match on pre-2005 averages
# ================================================================
print(f"\n{'#'*65}")
print(f"  PSM: MATCHING ON PRE-2005 CHARACTERISTICS")
print(f"{'#'*65}")

psm_vars = ['log_gdp_pc_current_usd','secondary_enroll_gross_pct',
            'percent_urban','birth_rate_crude_per_1000',
            'female_emp_ratio']

pre = (panel[panel['year'].between(2000,2004)]
       .groupby('Country Code')[psm_vars + ['treat_usaid']]
       .mean().reset_index().dropna())

print(f"Countries with pre-treatment data: {len(pre)}")

X       = StandardScaler().fit_transform(pre[psm_vars].values)
y       = pre['treat_usaid'].values
pscores = LogisticRegression(max_iter=1000,
                              random_state=42).fit(X, y).predict_proba(X)[:,1]
pre['pscore'] = pscores

# Nearest neighbor matching with caliper
treated = pre[pre['treat_usaid']==1].copy()
control = pre[pre['treat_usaid']==0].copy()
CALIPER = 0.05

matched = []
used_controls = set()
for _, t in treated.iterrows():
    avail = control[~control['Country Code'].isin(used_controls)]
    if len(avail) == 0:
        continue
    diffs  = abs(avail['pscore'] - t['pscore'])
    best   = diffs.idxmin()
    if diffs[best] <= CALIPER:
        matched.append((t['Country Code'], avail.loc[best,'Country Code']))
        used_controls.add(avail.loc[best,'Country Code'])

print(f"Matched pairs:   {len(matched)}")
print(f"Unmatched:       {len(treated) - len(matched)}")

matched_countries = set([t for t,c in matched] + [c for t,c in matched])
panel_m = panel[panel['Country Code'].isin(matched_countries)].copy()

# ── Balance check ──
print(f"\n  Balance check (post-matching):")
print(f"  {'Variable':<35} {'Treated':>8} {'Control':>8} {'p':>7}")
print(f"  {'-'*60}")
mt = pre[pre['Country Code'].isin([t for t,c in matched])]
mc = pre[pre['Country Code'].isin([c for t,c in matched])]
for v in psm_vars:
    _, p = scipy_stats.ttest_ind(mt[v].dropna(), mc[v].dropna())
    sig  = stars(p)
    print(f"  {v:<35} {mt[v].mean():>8.2f} {mc[v].mean():>8.2f} "
          f"{p:>7.3f} {sig}")

# ================================================================
# DiD ON MATCHED SAMPLE
# ================================================================
print(f"\n{'#'*65}")
print(f"  DiD ON MATCHED SAMPLE")
print(f"  (Compare to China main results)")
print(f"{'#'*65}")

outcomes = [
    ('primary_enroll_gross_pct',   'Primary Enrollment Gross (%)'),
    ('secondary_enroll_gross_pct', 'Secondary Enrollment Gross (%)'),
    ('secondary_gross_female',     'Secondary Gross Female (%)'),
    ('secondary_gross_male',       'Secondary Gross Male (%)'),
    ('tertiary_enroll_gross_pct',  'Tertiary Enrollment Gross (%)'),
    ('avg_years_schooling',        'Avg Years of Schooling'),
]

ssa_m = panel_m[panel_m['Region']=='Sub-Saharan Africa'].copy()

print(f"\n  {'Outcome':<35} {'Sample':<20} {'Coef':>8} {'SE':>8} "
      f"{'p':>7} {'N':>7} {'':>4}")
print(f"  {'-'*90}")

for ocol, olbl in outcomes:
    for sdf, slbl in [(panel_m, 'Full Matched'),
                      (ssa_m,   'SSA Matched')]:
        df = sdf.dropna(subset=[ocol,'Country Code','year',
                                  'treat_usaid','post_usaid',
                                  'log_gdp_pc_current_usd',
                                  'population_total','percent_urban',
                                  'birth_rate_crude_per_1000']).copy()
        df['did'] = df['treat_usaid'] * df['post_usaid']
        try:
            mod = smf.ols(
                f'Q("{ocol}") ~ did + {controls} + '
                f'C(year) + C(Q("Country Code"))',
                data=df
            ).fit(cov_type='cluster',
                  cov_kwds={'groups': df['Country Code']})
            c, se, p = (mod.params['did'], mod.bse['did'],
                        mod.pvalues['did'])
            print(f"  {olbl:<35} {slbl:<20} {c:>+8.3f} {se:>8.3f} "
                  f"{p:>7.3f} {int(mod.nobs):>7} {stars(p):>4}")
        except Exception as e:
            print(f"  {olbl:<35} {slbl:<20} ERROR: {e}")

print("\n\nDone.")
sys.stdout = _t
log.close()