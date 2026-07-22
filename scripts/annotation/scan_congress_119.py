"""Run the Stage 1 China keyword filter on Congress 119 ONLY.

Writes a separate manifest (china_filter_results_119.csv) so the existing
china_filter_results.csv (101-118) is not touched and ~270k bills are not
re-scanned.

Run from the project root:
    python scripts/annotation/scan_congress_119.py
"""
import sys
from pathlib import Path

# Put stage1/ directly on the path so `legislation_pipeline` and its internal
# `from constants import ...` both resolve regardless of package layout.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage1"))

from legislation_pipeline import ChinaLegislationPipeline, PipelineConfig  # noqa: E402


def main() -> None:
    config = PipelineConfig(
        raw_data_root=ROOT / "raw_data" / "raw_legislation",
        output_root=ROOT / "raw_data" / "china_legislation_119",  # unused (no stage2 call)
        csv_path=ROOT / "data" / "processed" / "china_filter_results_119.csv",
        congress_range=range(119, 120),
    )
    pipeline = ChinaLegislationPipeline(config)
    stats = pipeline.stage1()  # writes china_filter_results_119.csv; no stage2 copy
    print(f"Congress 119 survivors written: {stats.total_matched} matched")


if __name__ == "__main__":
    main()
