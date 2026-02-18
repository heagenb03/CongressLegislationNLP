from pathlib import Path
from legislation_pipeline import ChinaLegislationPipeline, PipelineConfig

CHINA_CONFIG = PipelineConfig(
    raw_data_root=Path("raw_data/raw_legislation"),
    output_root=Path("raw_data/china_legislation_101_118"),
    csv_path=Path("coded_data/china_filter_results.csv"),
    congress_range=range(101, 119),
)


def main() -> None:
    pipeline = ChinaLegislationPipeline(CHINA_CONFIG)

    #stats = pipeline.stage1()
    pipeline.stage2()


if __name__ == "__main__":
    main()