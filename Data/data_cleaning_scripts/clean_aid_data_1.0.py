import pandas as pd
import numpy as np

print("Loading CLG Global dataset...")
df = pd.read_excel('Data/BRI_Data/AidDatas_CLG_Global_Dataset_v1.0.xlsx',
                   sheet_name='CLG-Global 1.0_Records')

# ── Filter to recommended for aggregates only ──
df = df[df['Recommended_for_Aggregates']=='Yes'].copy()
df = df[df['ODA_Eligible_Recipient']=='Yes'].copy()
print(f"Records after ODA filter: {len(df):,}")
print(f"Records after filtering: {len(df):,}")

# ── Use Adjusted_Amount_Constant_USD_2023 as main amount ──
df['amount'] = df['Adjusted_Amount_Constant_USD_2023'].fillna(0)
df = df.rename(columns={
    'Country_of_Activity_ISO3': 'Country Code',
    'Commitment_Year':          'year',
    'Sector_Name':              'sector',
})

# ── Keep only 2000-2023 ──
df = df[(df['year']>=2000) & (df['year']<=2023)]

# ── Aggregate total investment by country-year ──
total = (df.groupby(['Country Code','year'])
           .agg(projects=('AidData_Record_ID','count'),
                usd_const_2023=('amount','sum'))
           .reset_index())
total['log1p_usd_const_2023'] = np.log1p(total['usd_const_2023'])

print(f"\nTotal aggregation:")
print(f"  Country-years: {len(total):,}")
print(f"  Countries: {total['Country Code'].nunique()}")
print(f"  Years: {total['year'].min()} - {total['year'].max()}")
print(f"  Mean investment per country-year: ${total['usd_const_2023'].mean()/1e6:.1f}M")

# ── Aggregate by sector ──
sectors = {
    'Education':    ['EDUCATION'],
    'Transport':    ['TRANSPORT AND STORAGE'],
    'Energy':       ['ENERGY'],
    'Health':       ['HEALTH'],
    'Industry':     ['INDUSTRY, MINING, CONSTRUCTION'],
}

sector_aggs = []
for label, sector_list in sectors.items():
    agg = (df[df['sector'].isin(sector_list)]
              .groupby(['Country Code','year'])
              .agg(**{f'usd_{label}': ('amount','sum'),
                      f'projects_{label}': ('AidData_Record_ID','count')})
              .reset_index())
    agg[f'log1p_{label}'] = np.log1p(agg[f'usd_{label}'])
    sector_aggs.append(agg)
    print(f"  {label}: {len(agg):,} country-years, "
          f"{agg['Country Code'].nunique()} countries")

# ── Merge all sectors into one wide file ──
sector_panel = sector_aggs[0]
for agg in sector_aggs[1:]:
    sector_panel = sector_panel.merge(agg, on=['Country Code','year'], how='outer')

# ── Merge total into sector panel ──
final = total.merge(sector_panel, on=['Country Code','year'], how='outer')

# ── Fill sector NaN with 0 (no investment = 0) ──
for label in sectors.keys():
    for col in [f'usd_{label}', f'projects_{label}', f'log1p_{label}']:
        if col in final.columns:
            final[col] = final[col].fillna(0)

# ── Build cumulative investment for threshold treatment ──
final = final.sort_values(['Country Code','year'])
final['cumul_usd'] = final.groupby('Country Code')['usd_const_2023'].cumsum()
final['cumul_usd_education'] = final.groupby('Country Code')[f'usd_Education'].cumsum()

print(f"\nFinal panel shape: {final.shape}")
print(f"Sample:")
print(final.head(5)[['Country Code','year','usd_const_2023','usd_Education',
                      'cumul_usd','log1p_usd_const_2023']].to_string())

# ── Save ──
final.to_csv('Data/clean_aid_data/aiddata_clg_country_year.csv', index=False)
print("\nSaved: Data/clean_aid_data/aiddata_clg_country_year.csv")

# ── Also build treatment indicators ──
# Main treatment: $500M cumulative total investment
crossed_500m = (final[final['cumul_usd'] >= 500e6]
                .groupby('Country Code')['year']
                .min().reset_index()
                .rename(columns={'year':'treat_year_500m'}))

# Education treatment: any cumulative education investment > $10M
crossed_edu = (final[final['cumul_usd_education'] >= 10e6]
               .groupby('Country Code')['year']
               .min().reset_index()
               .rename(columns={'year':'treat_year_edu'}))

print(f"\nTreatment summary:")
print(f"  $500M threshold: {len(crossed_500m)} countries")
print(f"  $10M education threshold: {len(crossed_edu)} countries")

treat = crossed_500m.merge(crossed_edu, on='Country Code', how='outer')
treat.to_csv('Data/clean_aid_data/treatment_indicators_clg.csv', index=False)
print("Saved: Data/clean_aid_data/treatment_indicators_clg.csv")