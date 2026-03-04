# clean_income_data.py
import pandas as pd
import numpy as np

df = pd.read_csv('Data/Raw_WBD/income_data.csv')

keep_series = {
    'GNI per capita, Atlas method (current US$)':                          'gni_per_capita',
    'GDP growth (annual %)':                                                'gdp_growth',
    'Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)': 'poverty_3day',
}

df = df[df['Series Name'].isin(keep_series.keys())].copy()
df['var_name'] = df['Series Name'].map(keep_series)

year_cols = [c for c in df.columns if 'YR' in c 
             and int(c[:4]) >= 2000 and int(c[:4]) <= 2021]

df_long = df.melt(
    id_vars=['Country Code','Country Name','var_name'],
    value_vars=year_cols,
    var_name='year_str',
    value_name='value'
)

df_long['year']  = df_long['year_str'].str[:4].astype(int)
df_long['value'] = pd.to_numeric(df_long['value'], errors='coerce')
df_long = df_long.drop(columns='year_str')

panel = df_long.pivot_table(
    index=['Country Code','Country Name','year'],
    columns='var_name',
    values='value'
).reset_index()
panel.columns.name = None

print("Income panel shape:", panel.shape)
print("Countries:", panel['Country Code'].nunique())
print("\nMissing values:")
for col in ['gni_per_capita','gdp_growth','poverty_3day']:
    miss = panel[col].isna().mean()*100
    print(f"  {col:<25} {miss:.1f}% missing")

panel.to_csv('Data/Panel/income_panel.csv', index=False)
print("\nSaved: Data/Panel/income_panel.csv")