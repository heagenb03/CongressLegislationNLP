"""No congress_nlp step subpackage may reach into another step's behavior.

Cross-step needs go through a leaf -- paths, splits, ids, rawdata -- or through
one of the stable read-only interfaces in ALLOWED_CROSS. This is what keeps the
re-run loops separable: changing the feature schema must not break packet
building, and a new classifier must not depend on how features are extracted.
"""

import re
from pathlib import Path

from congress_nlp import paths

SUBPACKAGES = ("filtering", "annotation", "features", "classifiers", "evaluation")

# Stable read-only interfaces, not step behavior:
#   filtering.keywords  - the keyword tables
#   features.load       - read features.csv
#   features.views      - name a model input
#   evaluation.metrics  - score predictions
# features.extract is deliberately absent: reading the table is allowed,
# rebuilding it is not.
ALLOWED_CROSS = {
    "congress_nlp.filtering.keywords",
    "congress_nlp.features.load",
    "congress_nlp.features.views",
    "congress_nlp.evaluation.metrics",
}

IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+(congress_nlp[\w.]*)", re.MULTILINE)


def test_no_subpackage_imports_another_subpackage():
    package_root = paths.PROJECT_ROOT / "congress_nlp"
    violations: list[str] = []

    for sub in SUBPACKAGES:
        for module in (package_root / sub).rglob("*.py"):
            source = module.read_text(encoding="utf-8")
            for imported in IMPORT_RE.findall(source):
                if imported in ALLOWED_CROSS:
                    continue
                parts = imported.split(".")
                if len(parts) < 2:
                    continue
                other = parts[1]
                if other in SUBPACKAGES and other != sub:
                    violations.append(f"{module.relative_to(package_root)} imports {imported}")

    assert not violations, "Cross-subpackage imports:\n" + "\n".join(violations)


def test_no_step_subpackage_imports_features_extract():
    """The specific regression this rule exists for.

    annotation/ used to import ID helpers from features/extract.py, which tied a
    sprint re-run to the feature schema. Kept as its own test so the reason
    survives even if ALLOWED_CROSS is edited later.
    """
    package_root = paths.PROJECT_ROOT / "congress_nlp"
    offenders = [
        str(module.relative_to(package_root))
        for sub in SUBPACKAGES
        if sub != "features"
        for module in (package_root / sub).rglob("*.py")
        if "congress_nlp.features.extract" in module.read_text(encoding="utf-8")
    ]
    assert not offenders, f"These import features.extract: {offenders}"
