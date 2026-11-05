#!/usr/bin/env python3
"""Step 1 -- fetch VSI-Bench, verify the archive layout, write task manifests.

    python scripts/01_prepare.py                      # all three source datasets
    python scripts/01_prepare.py --datasets scannetpp  # smoke test, ~1 GB

Nothing here needs a GPU. Safe to re-run; downloads and extractions are skipped
when already present.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsibench_eval import common as C  # noqa: E402

CHUNK = 1 << 22


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} ({dest.stat().st_size/1e9:.2f} GB)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  [get ] {url}")
    if shutil.which("curl"):
        subprocess.run(["curl", "-L", "--retry", "5", "--fail", "-o", str(tmp), url],
                       check=True)
    else:
        import urllib.request
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            total, seen = int(r.headers.get("content-length", 0)), 0
            while chunk := r.read(CHUNK):
                f.write(chunk)
                seen += len(chunk)
                if total:
                    print(f"\r        {seen/total*100:5.1f}%", end="", flush=True)
        print()
    tmp.rename(dest)
    print(f"  [ok  ] {dest.name} ({dest.stat().st_size/1e9:.2f} GB)")


def extract(zip_path: Path, target: Path) -> None:
    """The loader expects <VIDEO_DIR>/<dataset>/<scene>.mp4 -- the zips already
    contain that layout, so we extract straight into VIDEO_DIR."""
    target.mkdir(parents=True, exist_ok=True)
    print(f"  [unzip] {zip_path.name} -> {target}")
    if shutil.which("unzip"):
        subprocess.run(["unzip", "-q", "-o", str(zip_path), "-d", str(target)],
                       check=True)
    else:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(target)


def zip_complete(zip_path: Path, target: Path) -> bool:
    """True when every .mp4 in the archive already exists under target.

    This makes an interrupted unzip recoverable: a partially extracted dataset
    is not mistaken for a complete one just because its directory is non-empty.
    """
    with zipfile.ZipFile(zip_path) as z:
        members = {n: z.getinfo(n).file_size for n in z.namelist()
                   if n.lower().endswith(".mp4")}
    if not members:
        return False
    # Compare sizes, not just existence: two concurrent unzips into the same
    # directory leave truncated mp4s that exist but fail in ffmpeg.
    for name, size in members.items():
        path = target / name
        if not path.exists() or path.stat().st_size != size:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="scannet,arkitscenes,scannetpp")
    ap.add_argument("--tasks", default=",".join(C.TASKS))
    ap.add_argument("--no-download", action="store_true",
                    help="assume data is already in place; only verify + write manifests")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        if t not in C.TASKS:
            raise SystemExit(f"unknown task {t}; known: {list(C.TASKS)}")

    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(C.describe_paths())
    annotations = C.DATA_DIR / "test.jsonl"

    if not args.no_download:
        print("== annotations")
        fetch(f"{C.HF_REPO_URL}/test.jsonl", annotations)
        print("== videos")
        for ds in datasets:
            z = C.DATA_DIR / f"{ds}.zip"
            fetch(f"{C.HF_REPO_URL}/{ds}.zip", z)
            marker = C.VIDEO_DIR / ds
            if marker.exists() and zip_complete(z, C.VIDEO_DIR):
                print(f"  [skip] already extracted {marker}")
            else:
                extract(z, C.VIDEO_DIR)

    rows = C.read_jsonl(annotations)
    if not rows:
        raise SystemExit(f"no annotations at {annotations}")
    print(f"\nannotations: {len(rows)} questions, "
          f"{len({r['scene_name'] for r in rows})} scenes")

    # ---- verify the layout the official loader requires -------------------
    missing, per_task = [], {}
    for t in tasks:
        sel = [r for r in rows if r["question_type"] == t and r["dataset"] in datasets]
        per_task[t] = len(sel)
        for r in sel:
            if not C.video_path(r["dataset"], r["scene_name"]).exists():
                missing.append(str(C.video_path(r["dataset"], r["scene_name"])))
    if missing:
        uniq = sorted(set(missing))
        print(f"\n!! {len(uniq)} video files missing, e.g. {uniq[:3]}")
        print("   expected <VIDEO_DIR>/<dataset>/<scene>.mp4 under", C.VIDEO_DIR)
        return 1

    # ---- write manifests -------------------------------------------------
    C.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"source": C.HF_DATASET, "n_frames": C.N_FRAMES,
            "config_digest": C.config_digest(), "tasks": {}}
    for t in tasks:
        sel = [r for r in rows if r["question_type"] == t and r["dataset"] in datasets]
        if not sel:
            print(f"!! no rows for {t} in {datasets}")
            continue
        out = C.MANIFEST_DIR / f"{t}.jsonl"
        with open(out, "w") as f:
            for r in sel:
                f.write(json.dumps({
                    "id": r["id"], "task": t, "question_type": t,
                    "dataset": r["dataset"], "scene": r["scene_name"],
                    "question": r["question"], "options": r.get("options"),
                    "ground_truth": r["ground_truth"],
                }, ensure_ascii=False) + "\n")
        scenes = len({r["scene_name"] for r in sel})
        by_ds = {}
        for r in sel:
            by_ds[r["dataset"]] = by_ds.get(r["dataset"], 0) + 1
        meta["tasks"][t] = {"n": len(sel), "n_scenes": scenes, "by_dataset": by_ds,
                            "metric": C.TASKS[t]["metric"]}
        print(f"  {t:24s} n={len(sel):5d}  scenes={scenes:3d}  {by_ds}")

    (C.DATA_DIR / "manifest_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote manifests to {C.MANIFEST_DIR}")
    print(f"config digest: {C.config_digest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
