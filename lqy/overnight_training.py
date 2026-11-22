"""Run fixed source-transfer experiments and publish a reviewed Kaggle CSV."""

import argparse
import csv
import ctypes
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .cnn import read_json, run_lock
from detection.training import write_json


RUNS = Path("lqy/runs")
TEMPLATE = Path("ai-generated-image-detection-benchmark/sample_submission.csv")
RELEASED = RUNS / "codec-final" / "near-specialist" / "submission.csv"
KNOWN_CORRECT = 610
TARGET_CORRECT = 750
EXPERIMENTS = (("joint-01", "--joint-geometry"),
               ("altered-views-01", "--alter-both"),
               ("resampling-response-01", "--resample-response"),
               ("correlation-01", "--correlation"),
               ("joint-spectrum-01", "--joint-spectrum"),
               ("worst-profile-01", "--worst-profile"),
               ("correlation-full-01", "--test-source-tradeoff"))


def inspect_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        data = csv.DictReader(stream)
        if data.fieldnames != ["image_id", "label"]:
            raise ValueError(f"Unexpected columns in {path}")
        rows = list(data)
    if len(rows) != 1000 or len({row["image_id"] for row in rows}) != 1000:
        raise ValueError(f"Expected 1,000 unique IDs in {path}")
    if any(None in row or row["label"] not in {"0", "1"} for row in rows):
        raise ValueError(f"Unexpected CSV row in {path}")
    return [row["image_id"] for row in rows], [row["label"] for row in rows]


def assess_export(path, *, reference=RELEASED, known_correct=KNOWN_CORRECT):
    template, _ = inspect_csv(TEMPLATE)
    previous_ids, previous = inspect_csv(reference)
    ids, labels = inspect_csv(path)
    if ids != template or previous_ids != template:
        raise ValueError("Prediction IDs must match the template order")
    flips = sum(a != b for a, b in zip(previous, labels, strict=True))
    return {"prediction_changes": flips, "best_possible_correct": min(len(ids), known_correct + flips)}


def wait_for_joint(pilot, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            state = read_json(pilot / "summary.json")
        except (ValueError, FileNotFoundError):
            state = {}
        if state.get("stage") == "complete":
            return
        time.sleep(5)
    raise TimeoutError(f"The active joint source pilot has not completed at {pilot}")


def run(args, previous=None):
    report = previous or {"stage": "running", "experiments": {}}

    def record(**fields):
        report.update(fields)
        write_json(args.output / "summary.json", report)

    record(stage="running")
    for name, option in EXPERIMENTS:
        if name in report["experiments"]:
            continue
        pilot = RUNS / "canonical-augmentation" / ("correlation-01" if option == "--test-source-tradeoff" else name)
        full = RUNS / "codec-canonical" / name
        record(active=name)
        command = [sys.executable, "-B", "-u", "-m", "lqy.canonical_augmentation",
                   "--device", args.device, "--output", str(pilot), "--full-output", str(full)]
        if name == "joint-01" and not args.run_joint:
            wait_for_joint(pilot, 2100)
            command.append("--promote")
        elif option == "--test-source-tradeoff":
            command.extend(("--promote", option))
        else:
            command.append(option)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as error:
            report["experiments"][name] = {"stage": "error", "exit_code": error.returncode}
            record()
            continue
        evidence = read_json(pilot / "summary.json")
        experiment = {"pilot": str(pilot / "summary.json"), "source_gain": evidence["aggregate"]}
        report["experiments"][name] = experiment
        if (full / "summary.json").exists():
            full_state = read_json(full / "summary.json")
            experiment.update(full=str(full / "summary.json"), stage=full_state["stage"])
            submission = full / "submission.csv"
            if full_state["stage"] == "complete":
                checks = assess_export(submission)
                experiment["csv_checks"] = checks
                if checks["best_possible_correct"] >= TARGET_CORRECT:
                    staged = args.output / "submission.csv.tmp"
                    shutil.copyfile(submission, staged)
                    staged.replace(args.output / "submission.csv")
                    record(stage="candidate_ready", submission=str(args.output / "submission.csv"))
                    return
        else:
            experiment["stage"] = "source_pilot_fell_short"
        record()
    record(stage="experiments_complete", active=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--run-joint", action="store_true", help="Run the joint pilot instead of waiting for it")
    parser.add_argument("--resume-after", action="store_true", help="Continue with additional experiments after the current run")
    parser.add_argument("--after-experiment", help="Wait for this queued experiment before continuing")
    args = parser.parse_args()
    if args.after_experiment is not None and not args.resume_after:
        parser.error("--after-experiment requires --resume-after")
    keep_awake = ctypes.windll.kernel32.SetThreadExecutionState if sys.platform == "win32" else None
    if keep_awake is not None:
        keep_awake(0x80000001)
    try:
        previous = None
        if args.resume_after:
            deadline = time.monotonic() + 7200
            while time.monotonic() < deadline:
                previous = read_json(args.output / "summary.json")
                if previous["stage"] == "candidate_ready":
                    return
                prerequisite_finished = (args.after_experiment is None
                                         or args.after_experiment in previous["experiments"])
                if previous["stage"] == "experiments_complete" and prerequisite_finished:
                    break
                time.sleep(6)
            else:
                raise TimeoutError("The preceding experiments have not finished")
            # The status is written shortly before the main process releases its lock.
            time.sleep(3)
        with run_lock(args.output):
            run(args, previous)
    finally:
        if keep_awake is not None:
            keep_awake(0x80000000)


if __name__ == "__main__":
    main()
