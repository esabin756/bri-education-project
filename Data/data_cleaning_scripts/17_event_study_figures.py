import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

os.makedirs('Data/Figures', exist_ok=True)

panel = pd.read_csv('Data/Panel/panel_2023.csv')
aid   = pd.read_csv('Data/clean_aid_data/aiddata_clg_country_year.csv')

controls = ('log_gdp_pc_current_usd + population_total + '
            'percent_urban + birth_rate_crude_per_1000')
low_mid  = ['Low income','Lower middle income','Upper middle income']
ssa      = panel[panel['Region']=='Sub-Saharan Africa'].copy()

# ================================================================
# HELPERS
# ================================================================
def run_dynamic_main(df, outcome, n_lags=6, n_pre=4):
    df = df.copy()
    df = df.dropna(subset=['Country Code','year','treat_500m','treat_year',
                            'log_gdp_pc_current_usd','population_total',
                            'percent_urban','birth_rate_crude_per_1000',
                            outcome]).copy()
    df['rel_time'] = df['year'] - df['treat_year']
    for l in range(n_lags+1):
        df[f'lag{l}'] = ((df['treat_500m']==1) &
                          (df['rel_time']==l)).astype(float)
    for p in range(1, n_pre+1):
        df[f'pre{p}'] = ((df['treat_500m']==1) &
                          (df['rel_time']==-p)).astype(float)
    lag_terms = ' + '.join(
        [f'lag{l}' for l in range(n_lags+1)] +
        [f'pre{p}' for p in range(1, n_pre+1)]
    )
    mod = smf.ols(
        f'{outcome} ~ {lag_terms} + {controls} + '
        f'C(year) + C(Q("Country Code"))',
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['Country Code']})
    rows = []
    for pp in range(-n_pre, n_lags+1):
        col = f'pre{abs(pp)}' if pp < 0 else f'lag{pp}'
        if pp == -1:
            rows.append({'period': pp, 'coef': 0.0, 'se': 0.0, 'p': 1.0})
            continue
        rows.append({
            'period': pp,
            'coef':   mod.params[col],
            'se':     mod.bse[col],
            'p':      mod.pvalues[col],
        })
    return pd.DataFrame(rows), df['Country Code'].nunique(), int(mod.nobs)

def build_edu_treatment(df, threshold_m):
    df = df.copy()
    edu_cumul = (aid.sort_values(['Country Code','year'])
                    .assign(cumul_edu=lambda x: x.groupby('Country Code')
                                                  ['usd_Education'].cumsum()))
    crossed = (edu_cumul[edu_cumul['cumul_edu'] >= threshold_m * 1e6]
                        .groupby('Country Code')['year']
                        .min().reset_index()
                        .rename(columns={'year': 'treat_year_edu_thresh'}))
    df = df.drop(columns=[c for c in df.columns
                           if 'treat_year_edu_thresh' in c], errors='ignore')
    df = df.merge(crossed, on='Country Code', how='left')
    df['treat_edu_t'] = df['treat_year_edu_thresh'].notna().astype(int)
    df['post_edu_t']  = ((df['treat_edu_t']==1) &
                          (df['year'] >= df['treat_year_edu_thresh'])).astype(int)
    return df

def run_dynamic_edu(df, outcome, n_lags=6, n_pre=4):
    df = df.copy()
    df = df.dropna(subset=['Country Code','year','treat_edu_t',
                            'treat_year_edu_thresh',
                            'log_gdp_pc_current_usd','population_total',
                            'percent_urban','birth_rate_crude_per_1000',
                            outcome]).copy()
    df['rel_time'] = df['year'] - df['treat_year_edu_thresh']
    for l in range(n_lags+1):
        df[f'lag{l}'] = ((df['treat_edu_t']==1) &
                          (df['rel_time']==l)).astype(float)
    for p in range(1, n_pre+1):
        df[f'pre{p}'] = ((df['treat_edu_t']==1) &
                          (df['rel_time']==-p)).astype(float)
    lag_terms = ' + '.join(
        [f'lag{l}' for l in range(n_lags+1)] +
        [f'pre{p}' for p in range(1, n_pre+1)]
    )
    mod = smf.ols(
        f'{outcome} ~ {lag_terms} + {controls} + '
        f'C(year) + C(Q("Country Code"))',
        data=df
    ).fit(cov_type='cluster', cov_kwds={'groups': df['Country Code']})
    rows = []
    for pp in range(-n_pre, n_lags+1):
        col = f'pre{abs(pp)}' if pp < 0 else f'lag{pp}'
        if pp == -1:
            rows.append({'period': pp, 'coef': 0.0, 'se': 0.0, 'p': 1.0})
            continue
        rows.append({
            'period': pp,
            'coef':   mod.params[col],
            'se':     mod.bse[col],
            'p':      mod.pvalues[col],
        })
    return pd.DataFrame(rows), df['Country Code'].nunique(), int(mod.nobs)

# ================================================================
# BUILD ALL 6 DATASETS
# ================================================================
print("Running regressions for figures...")

# Panel 1: Full panel secondary enrollment — null result
dyn1, nc1, n1 = run_dynamic_main(panel, 'secondary_enroll_gross_pct')
print(f"  Panel 1 done: N={n1}")

# Panel 2: Full panel tertiary enrollment — null result
dyn2, nc2, n2 = run_dynamic_main(panel, 'tertiary_enroll_gross_pct')
print(f"  Panel 2 done: N={n2}")

# Panel 3: SSA secondary enrollment — main finding
dyn3, nc3, n3 = run_dynamic_main(ssa, 'secondary_enroll_gross_pct')
print(f"  Panel 3 done: N={n3}")

# Panel 4: SSA GDP growth — growth without development
dyn4, nc4, n4 = run_dynamic_main(ssa, 'gdp_growth')
print(f"  Panel 4 done: N={n4}")

# Panel 5: Full panel secondary $50M education treatment
df_edu50 = build_edu_treatment(panel, 50)
dyn5, nc5, n5 = run_dynamic_edu(df_edu50, 'secondary_enroll_gross_pct')
print(f"  Panel 5 done: N={n5}")

# Panel 6: SSA secondary $100M education treatment
df_ssa_edu100 = build_edu_treatment(ssa, 100)
dyn6, nc6, n6 = run_dynamic_edu(df_ssa_edu100, 'secondary_enroll_gross_pct')
print(f"  Panel 6 done: N={n6}")

# ================================================================
# PLOT
# ================================================================
print("Building figure...")

# Colors
COL_NULL = '#95A5A6'   # gray for null results
COL_NEG  = '#E74C3C'   # red for negative findings
COL_POS  = '#2E86AB'   # blue for positive findings
COL_EDU  = '#8E44AD'   # purple for education treatment

panels = [
    (dyn1, 'Full Panel: Secondary Enrollment\n(Gross %)',
     'Treatment: $500M Total Investment',
     COL_NULL, nc1, n1, 'null'),
    (dyn2, 'Full Panel: Tertiary Enrollment\n(Gross %)',
     'Treatment: $500M Total Investment',
     COL_NULL, nc2, n2, 'null'),
    (dyn3, 'Sub-Saharan Africa: Secondary Enrollment\n(Gross %)',
     'Treatment: $500M Total Investment',
     COL_NEG, nc3, n3, 'main'),
    (dyn4, 'Sub-Saharan Africa: GDP Growth\n(Annual %)',
     'Treatment: $500M Total Investment',
     COL_POS, nc4, n4, 'main'),
    (dyn5, 'Full Panel: Secondary Enrollment\n(Gross %)',
     'Treatment: $50M Education Investment',
     COL_EDU, nc5, n5, 'edu'),
    (dyn6, 'Sub-Saharan Africa: Secondary Enrollment\n(Gross %)',
     'Treatment: $100M Education Investment',
     COL_EDU, nc6, n6, 'edu'),
]

fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle(
    'Event Study: Chinese Development Finance and Development Outcomes\n'
    'TWFE DiD with Country and Year Fixed Effects | Clustered SE by Country',
    fontsize=13, fontweight='bold', y=0.98
)

for ax, (dyn, title, subtitle, color, nc, nobs, ptype) in \
        zip(axes.flatten(), panels):

    pre  = dyn[dyn['period'] <  0]
    post = dyn[dyn['period'] >= 0]

    # Shade regions
    ax.axvspan(-4.4, -0.5, alpha=0.04, color='gray',  zorder=0)
    ax.axvspan(-0.5,  6.4, alpha=0.04, color=color,   zorder=0)

    # Zero line and treatment line
    ax.axhline(0, color='black', linewidth=0.8,
               linestyle='--', alpha=0.5, zorder=1)
    ax.axvline(-0.5, color='gray', linewidth=1.2,
               linestyle=':', alpha=0.8, zorder=1)

    # Pre-treatment periods
    ax.errorbar(pre['period'], pre['coef'],
                yerr=1.96*pre['se'],
                fmt='o-', color='#7F8C8D',
                capsize=4, linewidth=2, markersize=6,
                label='Pre-treatment', zorder=3)

    # Post-treatment periods
    ax.errorbar(post['period'], post['coef'],
                yerr=1.96*post['se'],
                fmt='s-', color=color,
                capsize=4, linewidth=2.5, markersize=7,
                label='Post-treatment', zorder=3)

    # Mark significant post periods
    sig_post = post[post['p'] < 0.1]
    if len(sig_post) > 0:
        ax.scatter(sig_post['period'], sig_post['coef'],
                   s=120, color=color, zorder=4,
                   edgecolors='black', linewidth=1.5)

    # Labels
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlabel('Years Relative to Treatment', fontsize=9)
    ax.set_ylabel('Coefficient (percentage points)', fontsize=9)
    ax.set_xticks(range(-4, 7))
    ax.set_xticklabels([str(i) for i in range(-4, 7)], fontsize=8)

    # Treatment label
    ax.text(0.02, 0.97, subtitle,
            transform=ax.transAxes, fontsize=7.5,
            verticalalignment='top', color='gray',
            style='italic')

    # Sample size
    ax.text(0.98, 0.97, f'N={nobs:,} | {nc} countries',
            transform=ax.transAxes, fontsize=7.5,
            verticalalignment='top', horizontalalignment='right',
            color='gray')

    # Treatment arrow
    ax.text(-0.5, ax.get_ylim()[1]*0.92, ' Treatment',
            fontsize=7.5, color='gray', ha='left')

    # Row labels
    if ptype == 'null':
        ax.text(0.98, 0.05, 'NULL RESULT',
                transform=ax.transAxes, fontsize=8,
                horizontalalignment='right',
                color='gray', style='italic',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#F0F0F0', alpha=0.8))

    ax.legend(fontsize=8, loc='lower left')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Row annotations
for ax, label, color in [
    (axes[0,0], 'NULL RESULTS — No average effect of BRI investment', '#7F8C8D'),
    (axes[1,0], 'SSA HETEROGENEITY — Growth without human capital development', '#2C3E50'),
    (axes[2,0], 'EDUCATION INVESTMENT — Delayed negative enrollment effects', '#8E44AD'),
]:
    ax.annotate(label,
                xy=(-0.12, 0.5), xycoords='axes fraction',
                fontsize=8.5, fontweight='bold', color=color,
                rotation=90, ha='center', va='center')

plt.tight_layout(rect=[0.05, 0, 1, 0.97])
plt.savefig('Data/Figures/event_study_main.png',
            dpi=180, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Saved: Data/Figures/event_study_main.png")

# ================================================================
# FIGURE 2: SECTOR DOSE-RESPONSE BAR CHART
# ================================================================
print("Building sector figure...")

sector_vars = [
    ('log1p_Transport', 'Transport'),
    ('log1p_Energy',    'Energy'),
    ('log1p_Health',    'Health'),
    ('log1p_Industry',  'Industry'),
    ('log1p_Education', 'Education'),
]

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
fig2.suptitle(
    'Sector Dose-Response: Chinese Investment and Secondary Enrollment\n'
    'TWFE with Country and Year FE | Log Investment | Clustered SE by Country',
    fontsize=12, fontweight='bold'
)

for ax, (sdf, slbl) in zip(axes2, [
    (panel, 'Full Panel (N=251 countries)'),
    (ssa,   'Sub-Saharan Africa (N=48 countries)')
]):
    cols_needed = ['secondary_enroll_gross_pct','Country Code','year',
                   'log_gdp_pc_current_usd','population_total',
                   'percent_urban','birth_rate_crude_per_1000']
    df2 = sdf.dropna(subset=cols_needed).copy()

    coefs, ses, ps, labels = [], [], [], []
    for svar, slabel in sector_vars:
        try:
            mod = smf.ols(
                f'secondary_enroll_gross_pct ~ {svar} + {controls} + '
                f'C(year) + C(Q("Country Code"))',
                data=df2
            ).fit(cov_type='cluster',
                  cov_kwds={'groups': df2['Country Code']})
            coefs.append(mod.params[svar])
            ses.append(mod.bse[svar])
            ps.append(mod.pvalues[svar])
            labels.append(slabel)
        except:
            coefs.append(0); ses.append(0)
            ps.append(1); labels.append(slabel)

    colors = ['#2E86AB' if c > 0 else '#E74C3C' for c in coefs]
    bars = ax.bar(labels, coefs, color=colors, alpha=0.8,
                  width=0.5, zorder=2)
    ax.errorbar(labels, coefs, yerr=[1.96*s for s in ses],
                fmt='none', color='black', capsize=6,
                linewidth=1.5, zorder=3)
    ax.axhline(0, color='black', linewidth=0.8,
               linestyle='--', alpha=0.5)

    # Add significance stars
    for i, (c, p) in enumerate(zip(coefs, ps)):
        sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
        if sig:
            y_pos = c + 1.96*ses[i] + 0.002
            ax.text(i, y_pos, sig, ha='center',
                    fontsize=11, fontweight='bold', color='black')

    ax.set_title(slbl, fontsize=11, fontweight='bold')
    ax.set_ylabel('Coefficient on Log Investment\n(pp per log $M)', fontsize=9)
    ax.set_xlabel('Investment Sector', fontsize=9)
    ax.grid(True, alpha=0.25, axis='y', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend
    pos_patch = mpatches.Patch(color='#2E86AB', alpha=0.8, label='Positive effect')
    neg_patch = mpatches.Patch(color='#E74C3C', alpha=0.8, label='Negative effect')
    ax.legend(handles=[pos_patch, neg_patch], fontsize=8)

plt.tight_layout()
plt.savefig('Data/Figures/sector_dose_response.png',
            dpi=180, bbox_inches='tight',
            facecolor='white')
plt.close()
print("Saved: Data/Figures/sector_dose_response.png")
print("\nAll figures complete.")