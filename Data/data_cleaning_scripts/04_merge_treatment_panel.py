import pandas as pd

# ── Load files ──
panel = pd.read_csv('Data/Panel/panel_with_birthrate.csv')
aid_treat = pd.read_csv('Data/clean_aid_data/aiddata_with_treatment.csv')

# ── Keep only the treatment columns from aid ──
treat_cols = aid_treat[['Country Code', 'year', 'treat_year', 
                          'treat_500m', 'post_500m', 'rel_time',
                          'cumulative_inv']].copy()

# ── Drop old treatment cols from panel to avoid conflicts ──
panel = panel.drop(columns=['join_year', 'treated', 'post_join'], errors='ignore')

# ── Merge ──
panel = panel.merge(treat_cols, on=['Country Code', 'year'], how='left')

# ── Countries not in AidData get treat_500m = 0 (never-treated) ──
panel['treat_500m'] = panel['treat_500m'].fillna(0).astype(int)
panel['post_500m']  = panel['post_500m'].fillna(0).astype(int)
panel['cumulative_inv'] = panel['cumulative_inv'].fillna(0)

# ── Sanity check ──
print("Panel shape:", panel.shape)
print("Treated countries:", panel[panel['treat_500m']==1]['Country Code'].nunique())
print("Never-treated countries:", panel[panel['treat_500m']==0]['Country Code'].nunique())
print("Post-treatment obs:", panel['post_500m'].sum())
print("\nMissing primary enrollment:", panel['primary_enroll_gross_pct'].isna().sum())
print("Missing secondary enrollment:", panel['secondary_enroll_gross_pct'].isna().sum())

# ── Save ──
panel.to_csv('Data/Panel/panel_500m_treated.csv', index=False)
print("\nSaved.")