from pathlib import Path

from congress_nlp import paths


def test_project_root_is_the_repo_root_not_the_cwd():
    """Derived from __file__, so scripts run from any directory."""
    assert (paths.PROJECT_ROOT / "congress_nlp" / "paths.py").exists()
    assert (paths.PROJECT_ROOT / "requirements.txt").exists()


def test_every_exported_path_is_absolute():
    for name in (
        "RAW_LEGISLATION", "FILTERED_OUTPUT", "DATA_RAW", "GOLD_LABELS",
        "INTERN_DIR", "DATA_PROCESSED", "MANIFEST_DIR", "COVERAGE_CSV",
        "FEATURES_CSV", "ANNOTATION_DIR", "MODELS_DIR", "OUTPUTS_DIR",
    ):
        assert getattr(paths, name).is_absolute(), name


def test_intern_subdir_stays_relative_for_tmp_path_composition():
    """resolve_intern_files() joins this onto a caller-supplied root."""
    assert not paths.INTERN_SUBDIR.is_absolute()
    assert paths.INTERN_DIR == paths.PROJECT_ROOT / paths.INTERN_SUBDIR


def test_manifest_paths_globs_the_manifest_dir(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "china_filter_119.csv").write_text("", encoding="utf-8")
    (manifest_dir / "china_filter_101_118.csv").write_text("", encoding="utf-8")
    (manifest_dir / "notes.txt").write_text("", encoding="utf-8")
    monkeypatch.setattr(paths, "MANIFEST_DIR", manifest_dir)

    found = paths.manifest_paths()

    assert [p.name for p in found] == ["china_filter_101_118.csv", "china_filter_119.csv"]


def test_manifest_paths_returns_empty_when_dir_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "MANIFEST_DIR", tmp_path / "nope")
    assert paths.manifest_paths() == []
