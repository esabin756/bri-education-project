import pandas as pd
import numpy as np

df = pd.read_excel('Data/Raw_WBD/Net_Enrollment.xlsx', sheet_name='Data')

keep_series = {
    'School enrollment, primary (% net)':           'primary_net_pct',
    'School enrollment, primary, female (% net)':   'primary_net_female',
    'School enrollment, primary, male (% net)':     'primary_net_male',
    'School enrollment, secondary (% net)':         'secondary_net_pct',
    'School enrollment, secondary, female (% net)': 'secondary_net_female',
    'School enrollment, secondary, male (% net)':   'secondary_net_male',
    'School enrollment, tertiary (% gross)':        'tertiary_gross_pct',
    'School enrollment, tertiary, female (% gross)':'tertiary_gross_female',
    'School enrollment, tertiary, male (% gross)':  'tertiary_gross_male',
}

df = df[df['Series Name'].isin(keep_series.keys())].copy()
df['var_name'] = df['Series Name'].map(keep_series)

yr_cols = [c for c in df.columns if 'YR' in c
           and 2000 <= int(c[:4]) <= 2023]

df_long = df.melt(
    id_vars=['Country Code','var_name'],
    value_vars=yr_cols,
    var_name='year_str',
    value_name='value'
)
df_long['year']  = df_long['year_str'].str[:4].astype(int)
df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')

panel = df_long.pivot_table(
    index=['Country Code','year'],
    columns='var_name',
    values='value'
).reset_index()
panel.columns.name = None

print("Net enrollment panel shape:", panel.shape)
print("Countries:", panel['Country Code'].nunique())
print("Years:", panel['year'].min(), '-', panel['year'].max())
print("\nMissing values:")
for col in keep_series.values():
    if col in panel.columns:
        miss = panel[col].isna().mean()*100
        print(f"  {col:<30} {miss:.1f}% missing")

# ── Add gender gap variable ──
panel['secondary_gender_gap'] = (panel['secondary_net_female'] 
                                  - panel['secondary_net_male'])
panel['tertiary_gender_gap']  = (panel['tertiary_gross_female'] 
                                  - panel['tertiary_gross_male'])

print("\nGender gap (female - male):")
print(f"  Secondary mean gap: {panel['secondary_gender_gap'].mean():.2f}pp")
print(f"  Tertiary mean gap:  {panel['tertiary_gender_gap'].mean():.2f}pp")

panel.to_csv('Data/Panel/net_enrollment_panel.csv', index=False)
print("\nSaved: Data/Panel/net_enrollment_panel.csv")