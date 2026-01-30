import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1]   # .../Data
RAW_DIR = DATA_DIR / "Raw"
OUT_DIR = DATA_DIR / "Clean"
OUT_DIR.mkdir(exist_ok=True)

# ---- Load ----
path = RAW_DIR / "average_schooling_world.csv"
df = pd.read_csv(path)

# ---- Clean: wide -> long ----
# World Bank often uses ".." for missing
df = df.replace("..", pd.NA)

id_cols = ["Country Name", "Country Code"]
year_cols = [c for c in df.columns if "[YR" in c]  # e.g. "1960 [YR1960]"

long = df.melt(
    id_vars=id_cols,
    value_vars=year_cols,
    var_name="year",
    value_name="avg_schooling",
)

# Extract year number from "1960 [YR1960]"
long["year"] = long["year"].str.extract(r"(\d{4})").astype("Int64")

# Convert to numeric
long["avg_schooling"] = pd.to_numeric(long["avg_schooling"], errors="coerce")

# Drop missing values
long = long.dropna(subset=["avg_schooling", "year"])

# Remove exact duplicates (you seem to have duplicates like Afghanistan repeated)
long = long.drop_duplicates(subset=["Country Code", "year"])

# Optional: sort
long = long.sort_values(["Country Code", "year"]).reset_index(drop=True)

print("Clean shape:", long.shape)
print(long.head())

outpath = OUT_DIR / "average_schooling_world_clean.csv"
long.to_csv(outpath, index=False)
print("Saved:", outpath)