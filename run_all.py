"""Runs the full experiment set. Started for you by run.bat / run.sh.

You should not need to read or edit this file.

Everything it needs is inside this folder. It writes its output to
``results/experiments.jsonl`` and can be stopped and restarted at any time -
configurations that already finished are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
sys.path.insert(0, str(BUNDLE / "code"))

RESULTS_DIR = BUNDLE / "results"
RESULTS_FILE = RESULTS_DIR / "experiments.jsonl"
SEQ_DIR = BUNDLE / "data" / "aljubouri" / "sequences"
MMASD_SEQ_DIR = BUNDLE / "data" / "mmasd_plus" / "sequences"

MIN_PYTHON = (3, 10)
MIN_RAM_GB = 3.0
MIN_DISK_GB = 2.0

#: Host-machine path fragments that must never appear in a shipped text file.
FORBIDDEN_PATH_MARKERS = ("C:\\Users\\", "C:/Users/", "C:\\Work\\", "C:/Work/",
                          "/home/", "/Users/", "OneDrive")


# ===========================================================================
# plain-English helpers
# ===========================================================================
def say(msg: str = "") -> None:
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:      # legacy console code page
        print(msg.encode("ascii", "replace").decode("ascii"), flush=True)


def rule() -> None:
    say("-" * 68)


def human_time(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, m = seconds // 3600, (seconds % 3600) // 60
    if h and m:
        return f"{h} h {m} m"
    if h:
        return f"{h} h"
    if m:
        return f"{m} m"
    return f"{seconds} s"


def fail(title: str, detail: str, what_to_do: str) -> int:
    say()
    rule()
    say("SOMETHING WENT WRONG")
    rule()
    say(title)
    say()
    say("What to do:")
    say(f"  {what_to_do}")
    say()
    say("Please copy everything below and send it to Harshit:")
    say("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    say(detail.strip())
    say("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -")
    return 1


# ===========================================================================
# preflight
# ===========================================================================
def preflight(quiet: bool = False) -> tuple[bool, list[str], dict]:
    problems: list[str] = []
    info: dict = {}

    v = sys.version_info
    info["python"] = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) < MIN_PYTHON:
        problems.append(
            f"This computer is running Python {info['python']}, but this needs "
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer.\n"
            f"     Download it free from https://www.python.org/downloads/ "
            f"and install it, then run this again."
        )

    total, used, free = shutil.disk_usage(str(BUNDLE))
    info["free_disk_gb"] = round(free / 1e9, 1)
    if free / 1e9 < MIN_DISK_GB:
        problems.append(
            f"Only {info['free_disk_gb']} GB of free disk space; about "
            f"{MIN_DISK_GB} GB is needed.\n     Free up some space and run this again."
        )

    ram_gb = None
    try:
        if platform.system() == "Windows":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            st = MS()
            st.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            ram_gb = st.ullTotalPhys / 1e9
        else:
            ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:  # noqa: BLE001 - RAM detection is best-effort
        ram_gb = None
    info["ram_gb"] = round(ram_gb, 1) if ram_gb else "unknown"
    if ram_gb and ram_gb < MIN_RAM_GB:
        problems.append(
            f"This computer has about {info['ram_gb']} GB of memory; "
            f"{MIN_RAM_GB} GB is recommended.\n"
            f"     It may still work. Close other programs and try anyway."
        )

    info["cpu_cores"] = os.cpu_count() or 1
    try:
        import torch

        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception as exc:  # noqa: BLE001
        info["torch"] = f"NOT INSTALLED ({exc})"
        info["gpu"] = "unknown"
        problems.append(
            "The maths library (PyTorch) is missing.\n"
            "     Close this window, then double-click run.bat (Windows) or "
            "run.sh (Mac/Linux) again - it installs everything automatically."
        )

    if not SEQ_DIR.is_dir() or not list(SEQ_DIR.glob("*.npz")):
        problems.append(
            f"The data folder is missing or empty ({SEQ_DIR.name}).\n"
            "     The bundle may not have unzipped completely. Unzip it again, "
            "making sure you extract ALL files."
        )

    if not quiet:
        say("Checking this computer...")
        say(f"  Python version .... {info['python']}")
        say(f"  Processor cores ... {info['cpu_cores']}")
        say(f"  Memory ............ {info['ram_gb']} GB")
        say(f"  Free disk space ... {info['free_disk_gb']} GB")
        say(f"  Graphics card ..... {info.get('gpu', 'unknown')}")
        say(f"  Data files ........ {len(list(SEQ_DIR.glob('*.npz')))} found")
        say()
    return (not problems), problems, info


#: Directory names never scanned for host paths: these are created locally by
#: whoever runs the bundle (a virtualenv, caches, VCS metadata), so they are
#: expected to contain that machine's own paths - that is not a leak of the
#: machine that *built* the bundle, which is all this check is trying to catch.
SKIPPED_DIR_NAMES = {".venv", "venv", "env", "__pycache__", ".git", ".pytest_cache",
                     "node_modules", "site-packages"}


def _is_local_env_dir(d: Path) -> bool:
    """True for any directory that is itself a virtualenv (has pyvenv.cfg),
    regardless of what it happens to be named (`.venv`, `aut`, ...)."""
    return (d / "pyvenv.cfg").is_file()


def verify_no_host_paths() -> list[str]:
    """The bundle must not contain absolute paths from the machine that built it."""
    offenders: list[str] = []
    skip_dirs: list[Path] = []
    for d in BUNDLE.rglob("*"):
        if d.is_dir() and (d.name in SKIPPED_DIR_NAMES or _is_local_env_dir(d)):
            skip_dirs.append(d)
    for p in BUNDLE.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".py", ".md", ".txt", ".json",
                                                       ".yaml", ".yml", ".bat", ".sh", ".cfg"}:
            continue
        if p.name == "run_all.py":       # this file names the markers on purpose
            continue
        if any(sd in p.parents for sd in skip_dirs):
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for marker in FORBIDDEN_PATH_MARKERS:
            if marker in text:
                offenders.append(f"{p.relative_to(BUNDLE)} contains '{marker}'")
    return offenders


def verify_checksums() -> list[str]:
    manifest_path = BUNDLE / "manifest.json"
    if not manifest_path.is_file():
        return ["manifest.json is missing"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bad: list[str] = []
    for rel, expected in manifest.get("files", {}).items():
        f = BUNDLE / rel
        if not f.is_file():
            bad.append(f"missing file: {rel}")
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        if h != expected["sha256"]:
            bad.append(f"corrupted file: {rel}")
    return bad


# ===========================================================================
# experiment matrix
# ===========================================================================
def load_view(seq_dir: Path, name: str):
    import numpy as np

    z = np.load(seq_dir / f"{name}.npz", allow_pickle=True)
    return (z["values"], z["mask"], [str(s) for s in z["sample_ids"]],
            tuple(str(s) for s in z["names"]))


def build_matrix(seq_dir: Path, cohort: str = "full") -> list[dict]:
    """Jobs for one cohort.

    ``full``    - every recording.
    ``matched`` - the age- and sex-matched subsample. Only the headline
                  comparison is repeated there: the subsample is small, and
                  running the whole grid on it would invite over-reading.
    """
    views = sorted(p.stem for p in seq_dir.glob("*.npz"))
    base = [
        {"view": None, "model": "majority", "permuted": False, "cohort": cohort},
        {"view": None, "model": "confound_baseline", "permuted": False, "cohort": cohort},
        {"view": None, "model": "demographic_baseline", "permuted": False, "cohort": cohort},
    ]
    if cohort == "matched":
        views = [v for v in views if v.endswith("whole_body")]
    for v in views:
        for m in ("logreg_agg", "tcn", "gru"):
            base.append({"view": v, "model": m, "permuted": False, "cohort": cohort})
        base.append({"view": v, "model": "logreg_agg", "permuted": True, "cohort": cohort})
    return base


def job_key(dataset: str, task: str, job: dict, cfg: dict) -> str:
    payload = json.dumps(
        {"dataset": dataset, "task": task, "view": job["view"], "model": job["model"],
         "permuted": job["permuted"], "cohort": job.get("cohort", "full"), "cfg": cfg},
        sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


def completed_keys() -> set[str]:
    if not RESULTS_FILE.is_file():
        return set()
    keys = set()
    for line in RESULTS_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            keys.add(json.loads(line)["job_key"])
        except Exception:  # noqa: BLE001 - a truncated last line is survivable
            continue
    return keys


def run_one(dataset, task, job, cfg, full_meta, seq_dir, log_prefix):
    """Run one configuration over all outer folds; returns a result record."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    from asdmotor.evaluation import (aggregate_folds, assert_no_subject_leakage,
                                     classification_metrics, inner_folds, subject_folds)
    from asdmotor.models import (MODEL_REGISTRY, TrainConfig, count_parameters,
                                 predict_proba, train_model)
    from asdmotor.matching import demographic_features, fold_composition, match_on_age_sex
    from asdmotor.sequences import SequenceStandardiser, aggregate_statistics

    GRIDS = {
        "tcn": [{"channels": 24, "n_blocks": 2, "dropout": 0.3},
                {"channels": 48, "n_blocks": 3, "dropout": 0.5}],
        "gru": [{"hidden": 32, "n_layers": 1, "dropout": 0.3},
                {"hidden": 64, "n_layers": 1, "dropout": 0.5, "bidirectional": True}],
    }

    meta = full_meta
    cohort = job.get("cohort", "full")
    match_summary = None
    if cohort == "matched":
        res = match_on_age_sex(meta, tolerance_years=cfg["match_tolerance_years"])
        keep = res.index
        match_summary = res.summary
        meta = meta.iloc[keep].reset_index(drop=True)
    else:
        keep = np.arange(len(meta))

    y = meta["label"].to_numpy()
    groups = meta["subject_id"].to_numpy()
    labels = ["TD", "ASD"] if task == "asd_vs_td" else sorted(set(map(str, y)))
    n_classes = int(len(set(y.tolist())))

    cols: list[str] = []
    if job["view"] is None:
        values = np.zeros((len(meta), 2, 1), dtype=np.float32)
        mask = np.ones((len(meta), 2), dtype=bool)
        names = ("dummy",)
        if job["model"] == "confound_baseline":
            cols = [c for c in ("body_scale_m", "sequence_length") if c in meta.columns]
            conf = meta[cols].to_numpy(dtype=float)
        elif job["model"] == "demographic_baseline":
            conf, dn = demographic_features(meta)
            cols = list(dn)
    else:
        values, mask, ids, names = load_view(seq_dir, job["view"])
        if ids != list(full_meta["sample_id"]):
            raise RuntimeError(f"{job['view']}: sample order does not match metadata")
        values, mask = values[keep], mask[keep]

    rng = np.random.default_rng(cfg["seed"])
    y_used = rng.permutation(y) if job["permuted"] else y

    folds = subject_folds(y_used, groups, n_splits=cfg["n_splits"],
                          n_repeats=cfg["n_repeats"], seed=cfg["seed"])
    per_fold, selections = [], []
    t0 = time.perf_counter()

    for f in folds:
        assert_no_subject_leakage(groups, f.train_idx, f.test_idx)
        y_tr, y_te = y_used[f.train_idx], y_used[f.test_idx]

        if job["model"] == "majority":
            maj = int(np.bincount(y_tr, minlength=n_classes).argmax())
            pred = np.full(len(y_te), maj)
            proba = np.zeros((len(y_te), n_classes)); proba[:, maj] = 1.0
            info = {"n_parameters": 0, "train_seconds": 0.0}
        elif job["model"] in ("confound_baseline", "demographic_baseline"):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=4000, class_weight="balanced",
                                                   random_state=cfg["seed"]))
            ts = time.perf_counter()
            clf.fit(conf[f.train_idx], y_tr)
            info = {"n_parameters": int(clf[-1].coef_.size + clf[-1].intercept_.size),
                    "train_seconds": time.perf_counter() - ts}
            proba = np.zeros((len(y_te), n_classes))
            proba[:, clf.classes_] = clf.predict_proba(conf[f.test_idx])
            pred = proba.argmax(1)
        else:
            std = SequenceStandardiser().fit(values, mask, f.train_idx)
            xs = std.transform(values, mask)
            if job["model"] == "logreg_agg":
                a_tr, _ = aggregate_statistics(xs[f.train_idx], mask[f.train_idx], names)
                a_te, _ = aggregate_statistics(xs[f.test_idx], mask[f.test_idx], names)
                ts = time.perf_counter()
                clf = make_pipeline(StandardScaler(),
                                    LogisticRegression(max_iter=4000, class_weight="balanced",
                                                       random_state=cfg["seed"]))
                clf.fit(a_tr, y_tr)
                info = {"n_parameters": int(clf[-1].coef_.size + clf[-1].intercept_.size),
                        "train_seconds": time.perf_counter() - ts}
                proba = np.zeros((len(y_te), n_classes))
                proba[:, clf.classes_] = clf.predict_proba(a_te)
                pred = proba.argmax(1)
            else:
                best_hp, best = GRIDS[job["model"]][0], -np.inf
                if cfg["inner_search"]:
                    for hp in GRIDS[job["model"]]:
                        scores = []
                        for tr_i, va_i in inner_folds(y_used, groups, f.train_idx,
                                                      n_splits=cfg["inner_splits"],
                                                      seed=cfg["seed"]):
                            assert_no_subject_leakage(groups, tr_i, va_i, f.test_idx)
                            istd = SequenceStandardiser().fit(values, mask, tr_i)
                            ixs = istd.transform(values, mask)
                            mdl = MODEL_REGISTRY[job["model"]](values.shape[2], n_classes, **hp)
                            mdl, _ = train_model(mdl, ixs[tr_i], mask[tr_i], y_used[tr_i],
                                                 ixs[va_i], mask[va_i], y_used[va_i],
                                                 TrainConfig(seed=cfg["seed"],
                                                             epochs=cfg["inner_epochs"]))
                            p = predict_proba(mdl, ixs[va_i], mask[va_i]).argmax(1)
                            scores.append(float((p == y_used[va_i]).mean()))
                        s = float(np.mean(scores))
                        if s > best:
                            best_hp, best = hp, s
                selections.append(best_hp)
                mdl = MODEL_REGISTRY[job["model"]](values.shape[2], n_classes, **best_hp)
                ts = time.perf_counter()
                counts = np.bincount(y_tr, minlength=n_classes).astype(float)
                counts[counts == 0] = 1.0
                mdl, hist = train_model(
                    mdl, xs[f.train_idx], mask[f.train_idx], y_tr,
                    xs[f.train_idx], mask[f.train_idx], y_tr,
                    TrainConfig(seed=cfg["seed"], epochs=cfg["epochs"]),
                    class_weight=counts.sum() / (n_classes * counts))
                train_s = time.perf_counter() - ts
                ti = time.perf_counter()
                proba = predict_proba(mdl, xs[f.test_idx], mask[f.test_idx])
                inf_s = time.perf_counter() - ti
                pred = proba.argmax(1)
                info = {"n_parameters": count_parameters(mdl), "train_seconds": train_s,
                        "inference_ms_per_sample": 1000 * inf_s / max(len(y_te), 1),
                        **{k: v for k, v in hist.items() if isinstance(v, (int, float))}}

        m = classification_metrics(y_te, pred, proba, n_classes)
        m.update({k: v for k, v in info.items() if isinstance(v, (int, float))})
        m["repeat"], m["fold"] = f.repeat, f.fold
        m["n_train"], m["n_test"] = len(f.train_idx), len(f.test_idx)
        # Demographic composition of this split, recorded as scalars so that
        # imbalance stays visible per fold instead of being averaged away.
        m.update(fold_composition(meta, f.train_idx, f.test_idx))
        per_fold.append(m)

    agg = aggregate_folds(per_fold)
    fs, region = (job["view"].split("__") if job["view"] else ("none", "none"))
    return {
        "job_key": job_key(dataset, task, job, cfg),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": cohort, "matching": match_summary,
        "dataset": dataset, "task": task, "task_kind": (
            "primary_research_question" if task == "asd_vs_td" else "pipeline_validation_proxy"),
        "labels": labels, "feature_set": fs, "body_region": region, "view": job["view"],
        "model": job["model"], "permuted_label_control": job["permuted"],
        "n_samples": int(len(meta)), "n_subjects": int(meta["subject_id"].nunique()),
        "n_features": int(values.shape[2]) if job["view"] else len(cols),
        "sequence_length": int(values.shape[1]) if job["view"] else 0,
        "window_overlap": "none (one sequence per recording)",
        "normalisation": "per-fold z-score fitted on training subjects only",
        "split": f"repeated stratified group {cfg['n_splits']}-fold x {cfg['n_repeats']}, grouped by subject_id",
        "hyperparameter_selection": (
            f"inner group {cfg['inner_splits']}-fold over 2 candidates, equal budget for tcn and gru"
            if cfg["inner_search"] and job["model"] in ("tcn", "gru") else "fixed"),
        "selected_hyperparameters": selections, "seed": cfg["seed"],
        "wallclock_seconds": time.perf_counter() - t0,
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "config": cfg, "metrics": agg, "per_fold": per_fold,
    }


# ===========================================================================
# main
# ===========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--only-mmasd", action="store_true")
    args, _ = ap.parse_known_args()

    import pandas as pd
    import torch

    torch.set_num_threads(max(1, min(6, os.cpu_count() or 1)))

    say()
    rule()
    say("SMOKE TEST" if args.smoke else "MAIN RUN")
    rule()
    say()

    ok, problems, info = preflight()
    if not ok:
        return fail(
            "This computer is missing something the run needs.",
            "\n".join(f"  - {p}" for p in problems),
            "Follow the instructions above, then try again.")

    offenders = verify_no_host_paths()
    if offenders:
        return fail("The bundle contains paths from another computer.",
                    "\n".join(offenders),
                    "Send this message to Harshit - the bundle needs rebuilding.")

    bad = verify_checksums()
    if bad:
        return fail("Some files did not survive the download.",
                    "\n".join(bad),
                    "Delete the folder, unzip the bundle again, and rerun.")
    say("All files verified. Nothing is missing or damaged.")
    say()

    datasets: list[tuple[str, str, Path]] = []
    if not args.only_mmasd:
        datasets.append(("aljubouri_kinect", "asd_vs_td", SEQ_DIR))
    if MMASD_SEQ_DIR.is_dir() and list(MMASD_SEQ_DIR.glob("*.npz")):
        datasets.append(("mmasd_plus", "activity", MMASD_SEQ_DIR))
    elif not args.only_mmasd:
        say("Optional second dataset: not present. That is completely fine -")
        say("the main run does not need it. See GET_THE_OTHER_DATASET.md if you")
        say("were asked to add it.")
        say()

    cfg = {
        "n_splits": 5, "n_repeats": 1 if args.smoke else 2,
        "inner_splits": 2, "inner_search": not args.smoke,
        "inner_epochs": 20, "epochs": 10 if args.smoke else 40,
        "seed": args.seed,
        # Age tolerance for the matched subsample. 1 year is the value that
        # maximises matched n while closing the age gap to ~0.25 years.
        "match_tolerance_years": 1.0,
    }

    jobs: list[tuple] = []
    for dataset, task, seq_dir in datasets:
        meta = pd.read_csv(seq_dir / "sequence_metadata.csv", dtype={"subject_id": str})
        if "label" not in meta.columns:
            meta["label"] = (meta["person_index"] == 0).astype(int)
        matrix = build_matrix(seq_dir)
        if dataset == "aljubouri_kinect" and not args.smoke:
            # Repeat the headline comparison on the age- and sex-matched subsample.
            matrix += build_matrix(seq_dir, cohort="matched")
        if args.smoke:
            matrix = [j for j in matrix
                      if j["model"] in ("majority", "demographic_baseline",
                                        "logreg_agg", "tcn")
                      and (j["view"] in (None, "motor__whole_body"))
                      and not j["permuted"]][:4]
        for j in matrix:
            jobs.append((dataset, task, j, seq_dir, meta))

    done = completed_keys()
    todo = [j for j in jobs if job_key(j[0], j[1], j[2], cfg) not in done]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    say(f"There are {len(jobs)} things to run in total.")
    if done and not args.smoke:
        say(f"{len(jobs) - len(todo)} were already finished last time and will be skipped.")
    if not todo:
        say()
        say("Everything is already finished - nothing left to do.")
        say(f"Send this file back: {RESULTS_FILE}")
        return 0

    per_job = 25 if args.smoke else (240 if info["cpu_cores"] >= 8 else 420)
    say(f"About {len(todo)} left to do.")
    say(f"Estimated time: roughly {human_time(len(todo) * per_job)}.")
    say()
    say("You can use this computer normally while it runs.")
    say("You can close this window at any time - rerunning picks up where it left off.")
    say()
    rule()
    say()

    started = time.perf_counter()
    for n, (dataset, task, job, seq_dir, meta) in enumerate(todo, 1):
        elapsed = time.perf_counter() - started
        rate = elapsed / max(n - 1, 1) if n > 1 else per_job
        remaining = rate * (len(todo) - n + 1)
        what = job["view"] or job["model"]
        say(f"Run {n} of {len(todo)} - about {human_time(remaining)} left   ({what}, {job['model']})")
        try:
            rec = run_one(dataset, task, job, cfg, meta, seq_dir, what)
        except Exception:  # noqa: BLE001
            return fail(
                f"The run stopped at step {n} of {len(todo)}.",
                traceback.format_exc(),
                "Nothing already finished is lost. Send the message below to Harshit.")
        with RESULTS_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    say()
    rule()
    if args.smoke:
        say("Setup works - you can start the real run.")
        say("Close this window and double-click run.bat (Windows) or run.sh (Mac/Linux).")
    else:
        say("ALL DONE. Everything finished successfully.")
        say()
        say("Please send back this one file:")
        say()
        say(f"    {RESULTS_FILE}")
        say()
        say("It is inside the 'results' folder next to this file.")
    rule()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say()
        say("Stopped. Nothing already finished is lost - just run it again to continue.")
        sys.exit(0)
