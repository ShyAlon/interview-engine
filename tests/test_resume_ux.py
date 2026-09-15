"""UX: resuming from the wrong cwd must say where we looked and point at the
real session dir one level over (run-4 follow-up: he started in spec-generator/
and resumed from interview-engine/)."""
import sqlite3
from pathlib import Path

import pytest

from interview_engine import cli
from interview_engine.errors import UserFacingError
from test_cli_resume import run_cli_with_inputs


def test_missing_checkpoint_prints_absolute_path(monkeypatch, tmp_path):
    monkeypatch.setenv("VMODEL_ARTIFACTS_DIR",str(tmp_path/"artifacts")); monkeypatch.chdir(tmp_path)
    with pytest.raises(UserFacingError) as ei:
        run_cli_with_inputs([], ["prog", "--session", "20260908-181530-dead"])
    msg = ei.value.render()
    assert str((tmp_path / "artifacts" / "20260908-181530-dead" / "interview" / "checkpoints.sqlite").resolve()) in msg


def test_missing_checkpoint_finds_sibling_session(monkeypatch, tmp_path):
    # the real checkpoint lives one directory over, as when he started the run
    # from a different repo dir
    real = tmp_path / "spec-generator" / "out" / "20260908-181530-dead"
    real.mkdir(parents=True)
    sqlite3.connect(str(real / "checkpoints.sqlite")).close()
    wrong = tmp_path / "interview-engine"
    wrong.mkdir()
    monkeypatch.setenv("VMODEL_ARTIFACTS_DIR",str(tmp_path/"artifacts"))
    monkeypatch.chdir(wrong)
    with pytest.raises(UserFacingError) as ei:
        run_cli_with_inputs([], ["prog", "--session", "20260908-181530-dead"])
    assert "no saved state" in ei.value.render()
    assert (tmp_path / "artifacts" / "20260908-181530-dead" / "interview" / "checkpoints.sqlite").exists()


def test_missing_checkpoint_no_sibling_keeps_generic_hint(monkeypatch, tmp_path):
    monkeypatch.setenv("VMODEL_ARTIFACTS_DIR",str(tmp_path/"artifacts")); monkeypatch.chdir(tmp_path)
    with pytest.raises(UserFacingError) as ei:
        run_cli_with_inputs([], ["prog", "--session", "nope-nope"])
    assert "typo in the session id" in ei.value.render()


def test_wrong_cwd_resume_creates_no_spurious_outdir(monkeypatch, tmp_path):
    monkeypatch.setenv("VMODEL_ARTIFACTS_DIR",str(tmp_path/"artifacts")); monkeypatch.chdir(tmp_path)
    with pytest.raises(UserFacingError):
        run_cli_with_inputs([], ["prog", "--session", "20260908-181530-dead"])
    assert not (tmp_path / "out").exists(), "no out/ dir should be created before the check"
