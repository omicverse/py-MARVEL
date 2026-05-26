from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
EXTERNAL_ROOT = BENCHMARK_ROOT / "external"
RUNS_ROOT = BENCHMARK_ROOT / "runs"


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _copy_dir(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"missing required path: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _copy_file_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _command_record(name: str, argv: list[str], result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    return {
        "name": name,
        "argv": argv,
        "returncode": result.returncode,
    }


def _run_logged(name: str, argv: list[str], *, cwd: Path, log_dir: Path) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)
    (log_dir / f"{name}.stdout.log").write_text(result.stdout, encoding="utf-8")
    (log_dir / f"{name}.stderr.log").write_text(result.stderr, encoding="utf-8")
    record = _command_record(name, argv, result)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            argv,
            output=result.stdout,
            stderr=result.stderr,
        )
    return record


def _copy_code_snapshot(*, repo_root: Path, archive_dir: Path) -> None:
    code_dir = archive_dir / "code"
    _copy_dir(repo_root / "benchmark" / "scripts", code_dir / "benchmark_scripts")
    _copy_file_if_exists(repo_root / "pyproject.toml", code_dir / "pyproject.toml")
    _copy_file_if_exists(repo_root / "benchmark" / "README.md", code_dir / "benchmark_README.md")
    _copy_file_if_exists(repo_root / "benchmark" / "external" / "README.md", code_dir / "external_benchmark_README.md")

    r_root = repo_root.parent / "MARVEL"
    _copy_file_if_exists(r_root / "DESCRIPTION", code_dir / "MARVEL_DESCRIPTION")
    _copy_file_if_exists(r_root / "README.md", code_dir / "MARVEL_README.md")


def archive_existing_external_benchmark(
    *,
    repo_root: Path = REPO_ROOT,
    run_id: str | None = None,
    command_records: list[dict[str, Any]] | None = None,
) -> Path:
    run_id = run_id or _run_id()
    benchmark_root = repo_root / "benchmark"
    external_root = benchmark_root / "external"
    archive_dir = benchmark_root / "runs" / run_id

    if archive_dir.exists():
        raise FileExistsError(f"archive already exists: {archive_dir}")
    archive_dir.mkdir(parents=True)

    _copy_dir(external_root, archive_dir / "results")
    _copy_code_snapshot(repo_root=repo_root, archive_dir=archive_dir)
    (archive_dir / "logs").mkdir(exist_ok=True)

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "r_source_root": str(repo_root.parent / "MARVEL"),
        "results_dir": str(archive_dir / "results"),
        "code_snapshot_dir": str(archive_dir / "code"),
        "commands": command_records or [],
    }
    _write_json(archive_dir / "manifest.json", manifest)
    return archive_dir


def run_and_archive(*, repo_root: Path = REPO_ROOT, run_id: str | None = None) -> Path:
    run_id = run_id or _run_id()
    archive_dir = repo_root / "benchmark" / "runs" / run_id
    if archive_dir.exists():
        raise FileExistsError(f"archive already exists: {archive_dir}")

    log_dir = repo_root / "benchmark" / "runs" / f".{run_id}.logs.tmp"
    if log_dir.exists():
        shutil.rmtree(log_dir)
    log_dir.mkdir(parents=True)
    commands: list[dict[str, Any]] = []

    r_runs_dir = repo_root / "benchmark" / "external" / "r_runs"
    if r_runs_dir.exists():
        shutil.rmtree(r_runs_dir)

    commands.append(
        _run_logged(
            "build_external_benchmark",
            [sys.executable, str(repo_root / "benchmark" / "scripts" / "build_external_benchmark.py")],
            cwd=repo_root,
            log_dir=log_dir,
        )
    )
    commands.append(
        _run_logged(
            "plot_external_benchmark_report",
            [sys.executable, str(repo_root / "benchmark" / "scripts" / "plot_external_benchmark_report.py")],
            cwd=repo_root,
            log_dir=log_dir,
        )
    )

    final_archive = archive_existing_external_benchmark(
        repo_root=repo_root,
        run_id=run_id,
        command_records=commands,
    )
    _copy_dir(log_dir, final_archive / "logs")
    shutil.rmtree(log_dir)
    return final_archive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the external R/Python benchmark and archive code, logs, reports, and results."
    )
    parser.add_argument("--run-id", default=None, help="Archive directory name under benchmark/runs/")
    args = parser.parse_args()

    archive_dir = run_and_archive(run_id=args.run_id)
    print(archive_dir)


if __name__ == "__main__":
    main()
