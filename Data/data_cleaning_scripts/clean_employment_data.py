import pandas as pd
import numpy as np

df = pd.read_csv('Data/Raw_WBD/EmploymentData.csv')

# ── Keep only usable ILO modeled series ──
keep_series = {
    'Employment to population ratio, ages 15-24, total (%) (modeled ILO estimate)': 'youth_emp_ratio',
    'Employment to population ratio, 15+, female (%) (modeled ILO estimate)':        'female_emp_ratio',
    'Employment in industry (% of total employment) (modeled ILO estimate)':         'industry_emp_pct',
    'Unemployment, total (% of total labor force) (modeled ILO estimate)':           'unemployment_pct',
    'Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)': 'youth_unemp_pct',
    'Labor force, total':                                                             'labor_force_total',
}

df = df[df['Series Name'].isin(keep_series.keys())].copy()
df['var_name'] = df['Series Name'].map(keep_series)

# ── Reshape to long format ──
year_cols = [c for c in df.columns if 'YR' in c and int(c[0:4]) >= 2000 and int(c[0:4]) <= 2021]

df_long = df.melt(
    id_vars=['Country Code', 'Country Name', 'var_name'],
    value_vars=year_cols,
    var_name='year_str',
    value_name='value'
)

df_long['year'] = df_long['year_str'].str[:4].astype(int)
df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
df_long = df_long.drop(columns='year_str')

# ── Pivot to wide ──
panel = df_long.pivot_table(
    index=['Country Code', 'Country Name', 'year'],
    columns='var_name',
    values='value'
).reset_index()
panel.columns.name = None

print("Employment panel shape:", panel.shape)
print("Countries:", panel['Country Code'].nunique())
print("Years:", sorted(panel['year'].unique()))
print("\nMissing values:")
for col in ['youth_emp_ratio','female_emp_ratio','industry_emp_pct',
            'unemployment_pct','youth_unemp_pct','labor_force_total']:
    miss = panel[col].isna().mean()*100
    print(f"  {col:<30} {miss:.1f}% missing")

panel.to_csv('Data/Panel/employment_panel.csv', index=False)
print("\nSaved: Data/Panel/employment_panel.csv")