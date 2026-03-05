import pandas as pd
import numpy as np

print("Loading data...")
outcomes  = pd.read_csv('Data/Panel/outcomes_panel_2023.csv')
net_enroll= pd.read_csv('Data/Panel/net_enrollment_panel.csv')
aid       = pd.read_csv('Data/clean_aid_data/aiddata_clg_country_year.csv')
class_df  = pd.read_excel('Data/Raw_WBD/CLASS_2025_10_07.xlsx')

# ── Clean income classifications ──
class_df.columns = [str(c).strip() for c in class_df.columns]
class_df = class_df[['Economy','Code','Region','Income group']].copy()
class_df = class_df.rename(columns={
    'Code':         'Country Code',
    'Region':       'Region',
    'Income group': 'IncomeGroup'
})
class_df = class_df[class_df['Country Code'].notna()].copy()
class_df = class_df[class_df['Country Code'].str.len()==3].copy()

# ── Build full country-year skeleton 2000-2023 ──
countries = outcomes['Country Code'].unique()
years     = range(2000, 2024)
skeleton  = pd.MultiIndex.from_product(
    [countries, years], names=['Country Code','year']
).to_frame(index=False)

print(f"Skeleton: {len(skeleton):,} rows ({len(countries)} countries x 24 years)")

# ── Merge outcomes ──
panel = skeleton.merge(outcomes, on=['Country Code','year'], how='left')

# ── Merge net enrollment ──
panel = panel.merge(net_enroll, on=['Country Code','year'], how='left')

# ── Merge income classifications ──
panel = panel.merge(
    class_df[['Country Code','Region','IncomeGroup']],
    on='Country Code', how='left'
)

# ── Build treatment from new AidData ──
# Total investment treatment — $500M threshold
aid_sorted = aid.sort_values(['Country Code','year'])

crossed_500m = (aid[aid['cumul_usd'] >= 500e6]
                .groupby('Country Code')['year']
                .min().reset_index()
                .rename(columns={'year':'treat_year'}))

crossed_250m = (aid[aid['cumul_usd'] >= 250e6]
                .groupby('Country Code')['year']
                .min().reset_index()
                .rename(columns={'year':'treat_year_250m'}))

crossed_1b = (aid[aid['cumul_usd'] >= 1000e6]
              .groupby('Country Code')['year']
              .min().reset_index()
              .rename(columns={'year':'treat_year_1b'}))

# Education-specific treatment — $10M threshold
crossed_edu = (aid[aid['cumul_usd_education'] >= 10e6]
               .groupby('Country Code')['year']
               .min().reset_index()
               .rename(columns={'year':'treat_year_edu'}))

# Merge treatment years
panel = panel.merge(crossed_500m, on='Country Code', how='left')
panel = panel.merge(crossed_250m, on='Country Code', how='left')
panel = panel.merge(crossed_1b,   on='Country Code', how='left')
panel = panel.merge(crossed_edu,  on='Country Code', how='left')

# ── Build treat/post/rel_time indicators ──
# Main $500M
panel['treat_500m'] = panel['treat_year'].notna().astype(int)
panel['post_500m']  = ((panel['treat_500m']==1) &
                        (panel['year'] >= panel['treat_year'])).astype(int)
panel['rel_time']   = np.where(
    panel['treat_500m']==1,
    panel['year'] - panel['treat_year'],
    np.nan
)

# Education treatment
panel['treat_edu']  = panel['treat_year_edu'].notna().astype(int)
panel['post_edu']   = ((panel['treat_edu']==1) &
                        (panel['year'] >= panel['treat_year_edu'])).astype(int)
panel['rel_time_edu'] = np.where(
    panel['treat_edu']==1,
    panel['year'] - panel['treat_year_edu'],
    np.nan
)

# ── Merge sector investment ──
sector_cols = ['Country Code','year','log1p_usd_const_2023',
               'log1p_Education','log1p_Transport','log1p_Energy',
               'log1p_Health','log1p_Industry']
panel = panel.merge(
    aid[sector_cols],
    on=['Country Code','year'], how='left'
)
for col in sector_cols[2:]:
    panel[col] = panel[col].fillna(0)

print(f"\nFinal panel: {panel.shape}")
print(f"Countries: {panel['Country Code'].nunique()}")
print(f"Years: {panel['year'].min()} - {panel['year'].max()}")
print(f"\nTreatment summary:")
print(f"  $500M treated:     {panel[panel['treat_500m']==1]['Country Code'].nunique()}")
print(f"  Education treated: {panel[panel['treat_edu']==1]['Country Code'].nunique()}")
print(f"  Never treated:     {panel[panel['treat_500m']==0]['Country Code'].nunique()}")
print(f"\nPost-treatment obs: {panel['post_500m'].sum():,}")
print(f"\nMissing key variables:")
for col in ['primary_enroll_gross_pct','secondary_enroll_gross_pct',
            'secondary_net_pct','secondary_net_female','secondary_net_male',
            'log_gdp_pc_current_usd','female_emp_ratio','gdp_growth']:
    miss = panel[col].isna().mean()*100
    print(f"  {col:<35} {miss:.1f}% missing")

panel.to_csv('Data/Panel/panel_2023.csv', index=False)
print("\nSaved: Data/Panel/panel_2023.csv")