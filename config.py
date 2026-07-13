"""Path configuration for the SABR Mesoball park-effects rerun.

Every script imports paths from here; nothing else hard-codes a location.
Override the root with the SABR_MESOBALL_ROOT environment variable if the
project lives somewhere else (e.g., on another machine or a reviewer's clone).
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("SABR_MESOBALL_ROOT", r"C:\SABR_Mesoball"))

GITHUB_DIR = ROOT / "GitHub"
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "output"

# input files
PANEL = DATA_DIR / "pa_panel.parquet"
WOBA_WEIGHTS = DATA_DIR / "wOBA_weights.csv"
TEAMS = DATA_DIR / "Teams.csv"
BIOFILE = DATA_DIR / "biofile0.csv"
PITCHING = DATA_DIR / "pitching.csv"

# input directories
GAMELOG_DIR = DATA_DIR / "gamelogs"   # Retrosheet game logs, gl1910.txt .. gl2025.txt
WIND_DIR = DATA_DIR / "wind"          # NOAA ISD-lite station-year files

# optional inputs (warn, don't fail)
PARKCODE = DATA_DIR / "parkcode.txt"  # Retrosheet park code crosswalk

# large event-level file, streamed in place (Stage 8); override with
# SABR_PLAYS_CSV if it lives elsewhere
PLAYS_CSV = Path(os.environ.get(
    "SABR_PLAYS_CSV", r"C:\overnight_effect_data\retrosheet\plays.csv"))

EXPECTATIONS_DIR = GITHUB_DIR / "expectations"

# Seamheads Parkfactors database CSVs (Stage 4d cross-check). Licensed for
# individual research use only - NOT redistributed with this repository.
# Obtain from seamheads.com and point SABR_SEAMHEADS_DIR at the CSV folder.
SEAMHEADS_DIR = Path(os.environ.get(
    "SABR_SEAMHEADS_DIR", str(ROOT / "CSV Final Ballpark Files 2024")))
