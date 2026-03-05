import pandas as pd
import numpy as np

def clean_wdi_file(path, var_name):
    """For enrollment files with 4 header rows and plain year columns."""
    df = pd.read_csv(path, skiprows=4)
    df.columns = [str(c).strip('"').strip() for c in df.columns]
    year_cols = [c for c in df.columns if str(c).isdigit()
                 and 2000 <= int(c) <= 2023]
    keep_cols = ['Country Name', 'Country Code'] + year_cols
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[df['Country Code'].notna()].copy()
    df = df[df['Country Code'].str.len() == 3].copy()
    df = df[keep_cols]
    df_long = df.melt(
        id_vars=['Country Name', 'Country Code'],
        value_vars=year_cols,
        var_name='year',
        value_name=var_name
    )
    df_long['year'] = df_long['year'].astype(int)
    df_long[var_name] = pd.to_numeric(df_long[var_name], errors='coerce')
    return df_long[['Country Code', 'year', var_name]]

def clean_wdi_file_yr(path, var_name):
    """For files with YR-style year columns and no skiprows."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip('"').strip() for c in df.columns]
    year_cols = [c for c in df.columns if 'YR' in c
                 and 2000 <= int(c[:4]) <= 2023]
    keep_cols = ['Country Name', 'Country Code'] + year_cols
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[df['Country Code'].notna()].copy()
    df = df[df['Country Code'].str.len() == 3].copy()
    df = df[keep_cols]
    df_long = df.melt(
        id_vars=['Country Name', 'Country Code'],
        value_vars=year_cols,
        var_name='year_str',
        value_name=var_name
    )
    df_long['year'] = df_long['year_str'].str[:4].astype(int)
    df_long[var_name] = pd.to_numeric(df_long[var_name], errors='coerce')
    return df_long[['Country Code', 'year', var_name]]

# ── Enrollment ──
print("Cleaning enrollment files...")
primary   = clean_wdi_file('Data/Raw_WBD/Primary_School_enrollment.csv',
                            'primary_enroll_gross_pct')
secondary = clean_wdi_file('Data/Raw_WBD/Secondary_School_Enrollment.csv',
                            'secondary_enroll_gross_pct')
tertiary  = clean_wdi_file('Data/Raw_WBD/Tertiary_enrollment.csv',
                            'tertiary_enroll_gross_pct')

for name, df in [('Primary',primary),('Secondary',secondary),('Tertiary',tertiary)]:
    print(f"  {name:<12} {df['Country Code'].nunique()} countries, "
          f"{df['year'].min()}-{df['year'].max()}")

# ── Control variables ──
print("\nCleaning control variable files...")
gdp   = clean_wdi_file_yr('Data/Raw_WBD/GDP_Overtime_2.csv',
                           'gdp_pc_current_usd')
pop   = clean_wdi_file('Data/Raw_WBD/Total_population.csv',
                           'population_total')
urban = clean_wdi_file('Data/Raw_WBD/Percent_Urban.csv',
                           'percent_urban')
birth = clean_wdi_file_yr('Data/Raw_WBD/birthrate.csv',
                           'birth_rate_crude_per_1000')

# Log transform GDP
gdp['log_gdp_pc_current_usd'] = np.log(
    pd.to_numeric(gdp['gdp_pc_current_usd'], errors='coerce'))
gdp = gdp.drop(columns='gdp_pc_current_usd')

for name, df in [('GDP',gdp),('Population',pop),
                 ('Urban',urban),('Birth rate',birth)]:
    print(f"  {name:<12} {df['Country Code'].nunique()} countries, "
          f"{df['year'].min()}-{df['year'].max()}")

# ── Employment ──
print("\nCleaning employment file...")
emp_raw = pd.read_csv('Data/Raw_WBD/EmploymentData.csv')
keep_series = {
    'Employment to population ratio, ages 15-24, total (%) (modeled ILO estimate)': 'youth_emp_ratio',
    'Employment to population ratio, 15+, female (%) (modeled ILO estimate)':        'female_emp_ratio',
    'Employment in industry (% of total employment) (modeled ILO estimate)':         'industry_emp_pct',
    'Unemployment, total (% of total labor force) (modeled ILO estimate)':           'unemployment_pct',
    'Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)': 'youth_unemp_pct',
}
emp_raw = emp_raw[emp_raw['Series Name'].isin(keep_series.keys())].copy()
emp_raw['var_name'] = emp_raw['Series Name'].map(keep_series)
yr_cols = [c for c in emp_raw.columns if 'YR' in c
           and 2000 <= int(c[:4]) <= 2023]
emp_long = emp_raw.melt(
    id_vars=['Country Code','var_name'],
    value_vars=yr_cols,
    var_name='year_str', value_name='value')
emp_long['year']  = emp_long['year_str'].str[:4].astype(int)
emp_long['value'] = pd.to_numeric(emp_long['value'], errors='coerce')
emp_panel = emp_long.pivot_table(
    index=['Country Code','year'],
    columns='var_name', values='value').reset_index()
emp_panel.columns.name = None
print(f"  Employment: {emp_panel['Country Code'].nunique()} countries, "
      f"{emp_panel['year'].min()}-{emp_panel['year'].max()}")

# ── Income ──
print("\nCleaning income file...")
inc_raw = pd.read_csv('Data/Raw_WBD/income_data.csv')
keep_inc = {
    'GNI per capita, Atlas method (current US$)':                           'gni_per_capita',
    'GDP growth (annual %)':                                                 'gdp_growth',
    'Poverty headcount ratio at $3.00 a day (2021 PPP) (% of population)':  'poverty_3day',
}
inc_raw = inc_raw[inc_raw['Series Name'].isin(keep_inc.keys())].copy()
inc_raw['var_name'] = inc_raw['Series Name'].map(keep_inc)
yr_cols_inc = [c for c in inc_raw.columns if 'YR' in c
               and 2000 <= int(c[:4]) <= 2023]
inc_long = inc_raw.melt(
    id_vars=['Country Code','var_name'],
    value_vars=yr_cols_inc,
    var_name='year_str', value_name='value')
inc_long['year']  = inc_long['year_str'].str[:4].astype(int)
inc_long['value'] = pd.to_numeric(inc_long['value'], errors='coerce')
inc_panel = inc_long.pivot_table(
    index=['Country Code','year'],
    columns='var_name', values='value').reset_index()
inc_panel.columns.name = None
print(f"  Income: {inc_panel['Country Code'].nunique()} countries, "
      f"{inc_panel['year'].min()}-{inc_panel['year'].max()}")

# ── Political stability ──
print("\nCleaning political stability file...")
try:
    ps = clean_wdi_file_yr('Data/Raw_WBD/Political_Stability_Raw.csv',
                            'political_stability')
    print(f"  Political stability: {ps['Country Code'].nunique()} countries, "
          f"{ps['year'].min()}-{ps['year'].max()}")
except Exception as e:
    print(f"  Political stability error: {e}")
    ps = None

# ── Merge all into master panel ──
print("\nMerging into master panel...")
panel = primary.copy()
for df in [secondary, tertiary]:
    panel = panel.merge(df, on=['Country Code','year'], how='outer')

# Keep only rows with enrollment data
panel = panel[panel[['primary_enroll_gross_pct',
                      'secondary_enroll_gross_pct',
                      'tertiary_enroll_gross_pct']].notna().any(axis=1)]

# Merge controls
for df in [gdp, pop, urban, birth]:
    panel = panel.merge(df, on=['Country Code','year'], how='left')

# Merge employment and income
panel = panel.merge(emp_panel, on=['Country Code','year'], how='left')
panel = panel.merge(inc_panel, on=['Country Code','year'], how='left')

if ps is not None:
    panel = panel.merge(ps, on=['Country Code','year'], how='left')

print(f"\nFinal panel: {panel.shape}")
print(f"Countries: {panel['Country Code'].nunique()}")
print(f"Years: {panel['year'].min()} - {panel['year'].max()}")
print(f"\nMissing values:")
for col in panel.columns[2:]:
    miss = panel[col].isna().mean()*100
    if miss < 95:
        print(f"  {col:<35} {miss:.1f}% missing")

panel.to_csv('Data/Panel/outcomes_panel_2023.csv', index=False)
print("\nSaved: Data/Panel/outcomes_panel_2023.csv")

# ── Net enrollment ──
print("\nChecking net enrollment file...")
try:
    net = pd.read_excel('Data/Raw_WBD/Net_Enrollment.xlsx', skiprows=4)
    net.columns = [str(c).strip('"').strip() for c in net.columns]
    print("Columns:", net.columns.tolist()[:8])
    print("Shape:", net.shape)
    if 'Indicator Name' in net.columns:
        print("Series:", net['Indicator Name'].unique())
except Exception as e:
    print(f"Net enrollment error: {e}")