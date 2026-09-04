"""Every filesystem location the project reads or writes.

PROJECT_ROOT is derived from this file's location, not the working directory,
so scripts run correctly from anywhere.
"""

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

# Raw legislation JSON (gitignored, ~270k bills)
RAW_LEGISLATION: Path = PROJECT_ROOT / "raw_data" / "raw_legislation"
FILTERED_OUTPUT: Path = PROJECT_ROOT / "raw_data" / "china_legislation"

# Tracked source data
DATA_RAW: Path = PROJECT_ROOT / "data" / "raw"
GOLD_LABELS: Path = DATA_RAW / "twl_coded_legislation_101_to_118.csv"

# Relative so callers can join it onto a temp root in tests; INTERN_DIR is the
# absolute form used in production.
INTERN_SUBDIR: Path = Path("data") / "raw" / "Summer2026InternsData"
INTERN_DIR: Path = PROJECT_ROOT / INTERN_SUBDIR

# Generated artifacts
DATA_PROCESSED: Path = PROJECT_ROOT / "data" / "processed"
MANIFEST_DIR: Path = DATA_PROCESSED / "manifests"
COVERAGE_CSV: Path = DATA_PROCESSED / "filter_coverage_analysis.csv"
FEATURES_CSV: Path = DATA_PROCESSED / "features.csv"
ANNOTATION_DIR: Path = PROJECT_ROOT / "data" / "annotation"

MODELS_DIR: Path = PROJECT_ROOT / "models"
OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"

MANIFEST_GLOB = "china_filter_*.csv"


def manifest_paths() -> list[Path]:
    """Every Stage 1 manifest, sorted by name.

    Adding a congress means dropping a new china_filter_<label>.csv into
    MANIFEST_DIR; nothing downstream needs editing.
    """
    if not MANIFEST_DIR.exists():
        return []
    return sorted(MANIFEST_DIR.glob(MANIFEST_GLOB))
