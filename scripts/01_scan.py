"""Run the Stage 1 keyword filter over one congress or a range.

    python scripts/01_scan.py --congress 119
    python scripts/01_scan.py --congress-range 101-118 --copy
"""

import argparse

from congress_nlp import paths
from congress_nlp.filtering.pipeline import ChinaLegislationPipeline, PipelineConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--congress", type=int, help="Scan a single congress.")
    group.add_argument(
        "--congress-range",
        help="Scan an inclusive range, e.g. 101-118.",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Also copy matched data.json files into the filtered output tree.",
    )
    return parser.parse_args()


def resolve_range(args: argparse.Namespace) -> tuple[range, str]:
    """Return the congress range to scan and the label used in the manifest name."""
    if args.congress is not None:
        return range(args.congress, args.congress + 1), str(args.congress)
    low_str, _, high_str = args.congress_range.partition("-")
    low, high = int(low_str), int(high_str)
    return range(low, high + 1), f"{low}_{high}"


def main() -> None:
    args = parse_args()
    congress_range, label = resolve_range(args)

    paths.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(
        raw_data_root=paths.RAW_LEGISLATION,
        output_root=paths.FILTERED_OUTPUT,
        csv_path=paths.MANIFEST_DIR / f"china_filter_{label}.csv",
        congress_range=congress_range,
    )
    pipeline = ChinaLegislationPipeline(config)
    stats = pipeline.stage1()
    print(f"Matched {stats.total_matched} bills and {stats.total_amendments_matched} amendments.")

    if args.copy:
        pipeline.stage2()


if __name__ == "__main__":
    main()
