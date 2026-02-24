#!/workspaces/bri-education-project/.venv/bin/python
# -*- coding: utf-8 -*-
"""Clean the China Global Investment Tracker Excel data.

Run from the repository root with:

    python Data/GIT_Data/clean_git.py

or execute directly after giving the file execute permission.
"""

import pandas as pd
from pathlib import Path
import numpy as np


# Repo root (since this file is Data/GIT_Data/clean_git.py)
ROOT = Path(__file__).resolve().parents[2]

# SIMPLE: hardcode the exact location
INFILE = ROOT / "Data" / "GIT_Data" / "China-Global-Investment-Tracker-2025-Half.xlsx"
SHEET = "Dataset 1+2"

OUTDIR = ROOT / "Data" / "GIT_Data" / "clean_git"
OUTDIR.mkdir(parents=True, exist_ok=True)
OUTFILE = OUTDIR / "cgit_clean_transactions.csv"

if not INFILE.exists():
    raise FileNotFoundError(f"Can't find: {INFILE}\nRun: ls -la Data/GIT_Data")

# Read
cgit = pd.read_excel(INFILE, sheet_name=SHEET, header=4)
cgit.columns = cgit.columns.astype(str).str.strip()

# Rename only the important columns
cgit = cgit.rename(columns={
    "Year": "year",
    "Country": "country",
    "Quantity in Millions": "deal_musd",
    "Sector": "sector",
    "Subsector": "subsector",
    "Region": "region",
    "BRI": "bri_flag",
    "Greenfield": "greenfield",
})

# Clean types
cgit["year"] = pd.to_numeric(cgit["year"], errors="coerce")
cgit["deal_musd"] = pd.to_numeric(cgit["deal_musd"], errors="coerce")

# Make simple flags
cgit["bri_flag"] = pd.to_numeric(cgit.get("bri_flag", 0), errors="coerce").fillna(0).astype(int)
cgit["greenfield_flag"] = (cgit.get("greenfield", "").astype(str).str.strip().str.upper() == "G").astype(int)

# Drop junk rows + keep year range
cgit = cgit.dropna(subset=["year", "country"])
cgit = cgit[(cgit["year"] >= 1990) & (cgit["year"] <= 2025)].copy()

cgit.to_csv(OUTFILE, index=False)
print("Saved:", OUTFILE)
print(cgit.head())