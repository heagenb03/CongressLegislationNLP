"""

Two-stage OOP pipeline:
  Stage 1: scan raw_legislation JSON files for keywords, write CSV manifest.
  Stage 2: copy matched files to Spark input directory (reads CSV manifest).

Usage:
    python main.py

Fields searched per legislation item (all lowercased):
  Bills:      official_title, short_title, popular_title, titles[].title,
              summary.text, subjects[]
  Amendments: purpose, description, actions[].text
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

# Maps compact legislation type → dot-separated notation used in legislation_id
_LEGISLATION_TYPE_DOT: dict[str, str] = {
    "s":       "s",
    "hr":      "h.r",
    "sres":    "s.res",
    "hres":    "h.res",
    "sconres": "s.con.res",
    "hconres": "h.con.res",
    "sjres":   "s.j.res",
    "hjres":   "h.j.res",
    "samdt":   "s.amdt",
    "hamdt":   "h.amdt",
}

# Maps legislation type → category label
_LEGISLATION_CATEGORY: dict[str, str] = {
    "hr":      "bill",
    "s":       "bill",
    "hres":    "simple_res",
    "sres":    "simple_res",
    "hconres": "concurrent_res",
    "sconres": "concurrent_res",
    "hjres":   "joint_res",
    "sjres":   "joint_res",
    "hamdt":   "amendment",
    "samdt":   "amendment",
}


class LegislationResult(NamedTuple):
    congress: int
    legislation_type: str       # e.g. "hr", "s", "sres", "samdt", "hamdt"
    legislation_number: str     # e.g. "1151", "samdt1"
    legislation_id: str         # e.g. "101_s.1151", "108_s.amdt.1"
    src_path: Path
    matched: bool
    matched_keywords: tuple[str, ...]
    skipped: bool = False
    category: str = "bill"      # "bill", "simple_res", "concurrent_res", "joint_res", "amendment"


@dataclass
class FilterStats:
    total_examined: int = 0
    total_matched: int = 0
    total_skipped: int = 0
    total_amendments_examined: int = 0
    total_amendments_matched: int = 0
    by_congress: Counter[int] = field(default_factory=Counter)
    by_legislation_type: Counter[str] = field(default_factory=Counter)
    by_keyword: Counter[str] = field(default_factory=Counter)

    def record_match(self, result: LegislationResult) -> None:
        self.total_matched += 1
        self.by_congress[result.congress] += 1
        self.by_legislation_type[result.legislation_type] += 1
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
            "Collecting legislation paths for Congresses %d\u2013%d \u2026",
            self.config.congress_range.start,
            self.config.congress_range.stop - 1,
        )
        legislation_tasks = self._build_task_list()
        amendment_tasks = self._build_amendment_task_list()
        total = len(legislation_tasks) + len(amendment_tasks)

        if total == 0:
            logger.error(
                "No legislation files found under %s \u2014 check that raw_data_root is correct.",
                self.config.raw_data_root.resolve(),
            )
            return FilterStats()

        logger.info(
            "Found %d legislation items (%d bills/resolutions, %d amendments) across %d Congresses.",
            total,
            len(legislation_tasks),
            len(amendment_tasks),
            len(self.config.congress_range),
        )

        stats = FilterStats(
            total_examined=len(legislation_tasks),
            total_amendments_examined=len(amendment_tasks),
        )
        matched_results: list[LegislationResult] = []
        processed = 0

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures: dict = {}
            for task in legislation_tasks:
                futures[executor.submit(self._check_bill, *task)] = task
            for task in amendment_tasks:
                futures[executor.submit(self._check_amendment, *task)] = task

            for future in as_completed(futures):
                try:
                    result: LegislationResult = future.result()
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
                    if result.category == "amendment":
                        stats.total_amendments_matched += 1

        logger.info(
            "Keyword scan complete. %d matched (%d legislation, %d amendments).",
            stats.total_matched,
            stats.total_matched - stats.total_amendments_matched,
            stats.total_amendments_matched,
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
                rows.append((
                    int(row["congress"]),
                    row["legislation_type"],
                    row["legislation_number"],
                ))

        logger.info(
            "Copying %d matched legislation items to %s \u2026", len(rows), self.config.output_root
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

    def _extract_amendment_text(self, amendment: dict[str, Any]) -> str:
        """Concatenate all searchable text fields from an amendment record, lowercased."""
        parts: list[str] = []
        if purpose := amendment.get("purpose"):
            parts.append(purpose)
        if description := amendment.get("description"):
            parts.append(description)
        for action in amendment.get("actions", []):
            if isinstance(action, dict):
                if text := action.get("text"):
                    parts.append(text)
        return " ".join(parts).lower()

    def _find_matched_keywords(self, text: str) -> list[str]:
        """Return every keyword that appears as a whole word/phrase in text."""
        return [kw for kw, pat in self._patterns if pat.search(text)]

    @staticmethod
    def _make_legislation_id(congress: int, legislation_type: str, legislation_number: str) -> str:
        """Construct the canonical legislation_id from directory metadata."""
        dot_type = _LEGISLATION_TYPE_DOT.get(legislation_type, legislation_type)
        # Amendment dirs include type prefix (e.g., "samdt1") — extract digits only
        m = re.search(r"(\d+)$", legislation_number)
        number = m.group(1) if m else legislation_number
        return f"{congress}_{dot_type}.{number}"

    def _check_bill(
        self, congress: int, legislation_type: str, legislation_number: str, json_path: Path
    ) -> LegislationResult:
        """Load data.json and test for keyword presence."""
        category = _LEGISLATION_CATEGORY.get(legislation_type, "bill")
        legislation_id = self._make_legislation_id(congress, legislation_type, legislation_number)
        try:
            with json_path.open(encoding="utf-8") as fh:
                bill = json.load(fh)
            text = self._extract_searchable_text(bill)
            matched_kws = self._find_matched_keywords(text)
            matched = bool(matched_kws)
        except Exception as exc:
            logger.warning("Skipping %s \u2014 %s", json_path, exc)
            return LegislationResult(
                congress=congress,
                legislation_type=legislation_type,
                legislation_number=legislation_number,
                legislation_id=legislation_id,
                src_path=json_path,
                matched=False,
                matched_keywords=(),
                skipped=True,
                category=category,
            )
        return LegislationResult(
            congress=congress,
            legislation_type=legislation_type,
            legislation_number=legislation_number,
            legislation_id=legislation_id,
            src_path=json_path,
            matched=matched,
            matched_keywords=tuple(matched_kws),
            category=category,
        )

    def _check_amendment(
        self, congress: int, legislation_type: str, legislation_number: str, json_path: Path
    ) -> LegislationResult:
        """Load amendment data.json and test for keyword presence."""
        legislation_id = self._make_legislation_id(congress, legislation_type, legislation_number)
        try:
            with json_path.open(encoding="utf-8") as fh:
                amendment = json.load(fh)
            text = self._extract_amendment_text(amendment)
            matched_kws = self._find_matched_keywords(text)
            matched = bool(matched_kws)
        except Exception as exc:
            logger.warning("Skipping amendment %s \u2014 %s", json_path, exc)
            return LegislationResult(
                congress=congress,
                legislation_type=legislation_type,
                legislation_number=legislation_number,
                legislation_id=legislation_id,
                src_path=json_path,
                matched=False,
                matched_keywords=(),
                skipped=True,
                category="amendment",
            )
        return LegislationResult(
            congress=congress,
            legislation_type=legislation_type,
            legislation_number=legislation_number,
            legislation_id=legislation_id,
            src_path=json_path,
            matched=matched,
            matched_keywords=tuple(matched_kws),
            category="amendment",
        )

    def _build_task_list(self) -> list[tuple[int, str, str, Path]]:
        """Walk raw_data_root and collect (congress, type, number, path) tuples for bills."""
        tasks: list[tuple[int, str, str, Path]] = []
        for congress in self.config.congress_range:
            bills_root = self.config.raw_data_root / str(congress) / "bills"
            if not bills_root.exists():
                logger.warning("Missing congress directory: %s", bills_root)
                continue
            for legislation_type_dir in bills_root.iterdir():
                if not legislation_type_dir.is_dir():
                    continue
                for legislation_dir in legislation_type_dir.iterdir():
                    json_path = legislation_dir / "data.json"
                    if json_path.is_file():
                        tasks.append(
                            (congress, legislation_type_dir.name, legislation_dir.name, json_path)
                        )
        return tasks

    def _build_amendment_task_list(self) -> list[tuple[int, str, str, Path]]:
        """Walk raw_data_root and collect (congress, type, number, path) tuples for amendments."""
        tasks: list[tuple[int, str, str, Path]] = []
        for congress in self.config.congress_range:
            amendments_root = self.config.raw_data_root / str(congress) / "amendments"
            if not amendments_root.exists():
                continue  # Pre-108 congresses have no amendments directory
            for amdt_type_dir in amendments_root.iterdir():
                if not amdt_type_dir.is_dir():
                    continue
                for amdt_dir in amdt_type_dir.iterdir():
                    json_path = amdt_dir / "data.json"
                    if json_path.is_file():
                        tasks.append(
                            (congress, amdt_type_dir.name, amdt_dir.name, json_path)
                        )
        return tasks

    def _copy_to_output(
        self,
        congress: int,
        legislation_type: str,
        legislation_number: str,
    ) -> bool:
        """Copy one legislation item's data.json into the mirrored output directory. Returns True on success."""
        subdir = "amendments" if legislation_type in ("samdt", "hamdt") else "bills"
        src_path = (
            self.config.raw_data_root
            / str(congress) / subdir / legislation_type / legislation_number / "data.json"
        )
        dest = (
            self.config.output_root
            / str(congress) / subdir / legislation_type / legislation_number / "data.json"
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

    def _write_csv_manifest(self, matched_results: list[LegislationResult]) -> None:
        """Write a sorted CSV manifest of all matched legislation to config.csv_path."""
        self.config.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["legislation_id", "congress", "legislation_type", "legislation_number", "matched_keywords", "category"]
            )
            for r in sorted(matched_results, key=self._legislation_sort_key):
                writer.writerow([
                    r.legislation_id,
                    r.congress,
                    r.legislation_type,
                    r.legislation_number,
                    "|".join(r.matched_keywords),
                    r.category,
                ])
        logger.info("CSV manifest written to %s", self.config.csv_path.resolve())

    @staticmethod
    def _legislation_sort_key(r: LegislationResult) -> tuple[int, str, int, str]:
        """Sort key for CSV output: numeric legislation number within each congress/type."""
        m = re.search(r"(\d+)$", r.legislation_number)
        return (r.congress, r.legislation_type, int(m.group(1)) if m else 0, r.legislation_number)


class ChinaLegislationPipeline(LegislationPipeline):
    """Keyword pre-filter for China-related U.S. legislation."""

    keywords: list[str] = CHINA_KEYWORDS
    country_name: str = "china"
