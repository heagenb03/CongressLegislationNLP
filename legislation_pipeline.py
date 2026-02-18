"""

Two-stage OOP pipeline:
  Stage 1: scan raw_legislation JSON files for keywords, write CSV manifest.
  Stage 2: copy matched files to Spark input directory (reads CSV manifest).

Usage:
    python main.py

Fields searched per bill (all lowercased):
    official_title, short_title, popular_title, titles[].title,
    summary.text, subjects[]
"""
import csv
import json
import logging
import re
import shutil
from abc import ABC, abstractmethod
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple
from constants import CHINA_KEYWORDS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class BillResult(NamedTuple):
    congress: int
    bill_type: str
    bill_number: str
    bill_id: str
    src_path: Path
    matched: bool
    matched_keywords: tuple[str, ...]
    skipped: bool = False 


@dataclass
class FilterStats:
    total_examined: int = 0
    total_matched: int = 0
    total_skipped: int = 0
    by_congress: Counter[int] = field(default_factory=Counter)
    by_bill_type: Counter[str] = field(default_factory=Counter)
    by_keyword: Counter[str] = field(default_factory=Counter)

    def record_match(self, result: BillResult) -> None:
        self.total_matched += 1
        self.by_congress[result.congress] += 1
        self.by_bill_type[result.bill_type] += 1
        for kw in result.matched_keywords:
            self.by_keyword[kw] += 1


@dataclass(frozen=True)
class PipelineConfig:
    raw_data_root: Path
    output_root: Path
    csv_path: Path
    congress_range: range
    max_workers: int = 8


class LegislationPipeline(ABC):
    """Country-agnostic base pipeline. Subclass and define `keywords` and `country_name`."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._patterns = self._compile_patterns()

    @property
    @abstractmethod
    def keywords(self) -> list[str]: ...

    @property
    @abstractmethod
    def country_name(self) -> str: ...

    def stage1(self) -> FilterStats:
        """Keyword scan + CSV manifest. Does NOT copy files."""
        logger.info(
            "Collecting bill paths for Congresses %d\u2013%d \u2026",
            self.config.congress_range.start,
            self.config.congress_range.stop - 1,
        )
        tasks = self._build_task_list()
        total = len(tasks)

        if total == 0:
            logger.error(
                "No bill files found under %s \u2014 check that raw_data_root is correct.",
                self.config.raw_data_root.resolve(),
            )
            return FilterStats()

        logger.info(
            "Found %d bills to examine across %d Congresses.",
            total,
            len(self.config.congress_range),
        )

        stats = FilterStats(total_examined=total)
        matched_results: list[BillResult] = []
        processed = 0

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._check_bill, *task): task for task in tasks
            }
            for future in as_completed(futures):
                try:
                    result: BillResult = future.result()
                except Exception as exc:
                    task = futures[future]
                    logger.warning("Unexpected error processing %s \u2014 %s", task[3], exc)
                    processed += 1
                    continue

                processed += 1
                if processed % 25_000 == 0:
                    pct = 100.0 * processed / total
                    logger.info("  \u2026 %d / %d (%.0f%%) examined", processed, total, pct)
                if result.skipped:
                    stats.total_skipped += 1
                elif result.matched:
                    matched_results.append(result)
                    stats.record_match(result)

        logger.info(
            "Keyword scan complete. %d / %d bills matched.", stats.total_matched, total
        )
        self._write_csv_manifest(matched_results)
        return stats

    def stage2(self) -> None:
        """Copy matched files from CSV manifest to output directory."""
        if not self.config.csv_path.exists():
            raise FileNotFoundError(
                f"CSV manifest not found: {self.config.csv_path.resolve()}\n"
                "Run stage1() first to generate the manifest."
            )

        rows: list[tuple[int, str, str]] = []
        with self.config.csv_path.open(encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append((int(row["congress"]), row["bill_type"], row["bill_number"]))

        logger.info(
            "Copying %d matched bills to %s \u2026", len(rows), self.config.output_root
        )
        self.config.output_root.mkdir(parents=True, exist_ok=True)

        copied = 0
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = [executor.submit(self._copy_to_output, *row) for row in rows]
            for future in as_completed(futures):
                if future.result():
                    copied += 1

        failed = len(rows) - copied
        logger.info("stage2 complete \u2014 %d / %d files copied (%d failed).", copied, len(rows), failed)

    def _compile_patterns(self) -> list[tuple[str, re.Pattern[str]]]:
        return [
            (kw, re.compile(r"\b" + re.escape(kw) + r"\b"))
            for kw in self.keywords
        ]

    def _extract_searchable_text(self, bill: dict[str, Any]) -> str:
        """Concatenate all searchable text fields from a bill record, lowercased."""
        parts: list[str] = []

        for key in ("official_title", "short_title", "popular_title"):
            if value := bill.get(key):
                parts.append(value)

        for title_entry in bill.get("titles", []):
            if text := title_entry.get("title"):
                parts.append(text)

        summary = bill.get("summary")
        if isinstance(summary, dict):
            if text := summary.get("text"):
                parts.append(text)

        for subject in bill.get("subjects", []):
            if isinstance(subject, str):
                parts.append(subject)
            elif isinstance(subject, dict) and (name := subject.get("name")):
                # Some schema versions use {"name": "Foreign Affairs", ...}
                parts.append(name)

        return " ".join(parts).lower()

    def _find_matched_keywords(self, text: str) -> list[str]:
        """Return every keyword that appears as a whole word/phrase in text."""
        return [kw for kw, pat in self._patterns if pat.search(text)]

    def _check_bill(
        self, congress: int, bill_type: str, bill_number: str, json_path: Path
    ) -> BillResult:
        """Load data.json and test for keyword presence."""
        try:
            with json_path.open(encoding="utf-8") as fh:
                bill = json.load(fh)
            bill_id = bill.get("bill_id", f"{bill_type}{bill_number}-{congress}")
            text = self._extract_searchable_text(bill)
            matched_kws = self._find_matched_keywords(text)
            matched = bool(matched_kws)
        except Exception as exc:
            logger.warning("Skipping %s \u2014 %s", json_path, exc)
            return BillResult(
                congress=congress,
                bill_type=bill_type,
                bill_number=bill_number,
                bill_id=f"{bill_type}{bill_number}-{congress}",
                src_path=json_path,
                matched=False,
                matched_keywords=(),
                skipped=True,
            )

        return BillResult(
            congress=congress,
            bill_type=bill_type,
            bill_number=bill_number,
            bill_id=bill_id,
            src_path=json_path,
            matched=matched,
            matched_keywords=tuple(matched_kws),
        )

    def _build_task_list(self) -> list[tuple[int, str, str, Path]]:
        """Walk raw_data_root and collect (congress, type, number, path) tuples."""
        tasks: list[tuple[int, str, str, Path]] = []
        for congress in self.config.congress_range:
            bills_root = self.config.raw_data_root / str(congress) / "bills"
            if not bills_root.exists():
                logger.warning("Missing congress directory: %s", bills_root)
                continue
            for bill_type_dir in bills_root.iterdir():
                if not bill_type_dir.is_dir():
                    continue
                for bill_dir in bill_type_dir.iterdir():
                    json_path = bill_dir / "data.json"
                    if json_path.is_file():
                        tasks.append(
                            (congress, bill_type_dir.name, bill_dir.name, json_path)
                        )
        return tasks

    def _copy_to_output(self, congress: int, bill_type: str, bill_number: str) -> bool:
        """Copy one bill's data.json into the mirrored output directory. Returns True on success."""
        src_path = (
            self.config.raw_data_root
            / str(congress)
            / "bills"
            / bill_type
            / bill_number
            / "data.json"
        )
        dest = (
            self.config.output_root
            / str(congress)
            / "bills"
            / bill_type
            / bill_number
            / "data.json"
        )
        if not src_path.exists():
            logger.warning("Source not found: %s", src_path)
            return False
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest)
            return True
        except OSError as exc:
            logger.warning("Failed to copy %s \u2014 %s", src_path, exc)
            return False

    def _write_csv_manifest(self, matched_results: list[BillResult]) -> None:
        """Write a sorted CSV manifest of all matched bills to config.csv_path."""
        self.config.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["bill_id", "congress", "bill_type", "bill_number", "matched_keywords"]
            )
            for r in sorted(matched_results, key=self._bill_sort_key):
                writer.writerow([
                    r.bill_id,
                    r.congress,
                    r.bill_type,
                    r.bill_number,
                    "|".join(r.matched_keywords),
                ])
        logger.info("CSV manifest written to %s", self.config.csv_path.resolve())

    @staticmethod
    def _bill_sort_key(r: BillResult) -> tuple[int, str, int, str]:
        """Sort key for CSV output: numeric bill number within each congress/type."""
        m = re.search(r"(\d+)$", r.bill_number)
        return (r.congress, r.bill_type, int(m.group(1)) if m else 0, r.bill_number)


class ChinaLegislationPipeline(LegislationPipeline):
    """Keyword pre-filter for China-related U.S. legislation."""

    keywords: list[str] = CHINA_KEYWORDS
    country_name: str = "china"
