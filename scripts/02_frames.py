#!/usr/bin/env python3
"""Step 2 -- decode and cache exactly N_FRAMES frames per scene, deterministically.

Frames are picked at `round(linspace(0, n_frames_total-1, N_FRAMES))` over the
decoded frame count, capped at FRAME_MAX_SIDE on the long side, and written as
JPEGs plus a sidecar `frames.json` recording the source frame count and the
exact chosen indices. Those indices go into the report and into every result
row, because "which 32 frames" is part of the protocol.

Uses ffmpeg's `select` filter in a single pass (no cv2/decord dependency, and
deterministic -- no fps resampling drift).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vsibench_eval import common as C  # noqa: E402


def probe(path: Path) -> tuple[int, float]:
    """(n_frames, fps) via ffprobe, falling back to duration*fps."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets,r_frame_rate,duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    st = json.loads(out)["streams"][0]
    num, den = (st.get("r_frame_rate") or "30/1").split("/")
    fps = float(num) / float(den or 1)
    n = int(st.get("nb_read_packets") or 0)
    if n <= 0:
        dur = float(st.get("duration") or 0)
        n = int(round(dur * fps))
    return n, fps


def extract(path: Path, outdir: Path, n_want: int, max_side: int) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    # Per-scene lock. Two jobs extracting into the same directory delete each
    # other's JPEGs mid-run and produce "SHORT" scenes and rename errors that
    # look like ffmpeg faults. A resubmitted data job is exactly how that
    # happens, so make the second one fail loudly instead of corrupting.
    lock = outdir / ".extract.lock"
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError(
            "another process is extracting this scene (.extract.lock present); "
            "is a duplicate data job running?") from None
    try:
        return _extract_locked(path, outdir, n_want, max_side)
    finally:
        try:
            lock.rmdir()
        except OSError:
            pass


def _extract_locked(path: Path, outdir: Path, n_want: int, max_side: int) -> dict:
    # Invalidate the old marker before touching the JPEGs.  If ffmpeg fails,
    # the next run must not trust a stale frames.json (which could claim 32
    # frames while the directory is empty).
    marker = outdir / "frames.json"
    if marker.exists():
        marker.unlink()
    for f in outdir.glob("*.jpg"):
        f.unlink()

    total, fps = probe(path)
    if total <= 0:
        raise RuntimeError(f"{path}: could not determine frame count")
    idx = sorted({int(round(x)) for x in
                  __import__("numpy").linspace(0, total - 1, n_want)})
    expr = "+".join(f"eq(n\\,{i})" for i in idx)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf",
         f"select='{expr}',scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'",
         "-vsync", "0", "-q:v", "3", str(outdir / "%03d.jpg")],
        check=True)
    got = sorted(outdir.glob("*.jpg"))
    if not got:
        raise RuntimeError(f"{path}: ffmpeg produced no frames")
    for i, f in enumerate(got):                     # normalise to 000..031
        f.rename(outdir / f"{i:03d}.jpg")
    meta = {"video": str(path), "source_frames": total, "fps": round(fps, 3),
            "n_requested": n_want, "n_extracted": len(got),
            "source_frame_indices": idx[:len(got)],
            "max_side": max_side}
    (outdir / "frames.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=",".join(C.TASKS))
    ap.add_argument("--limit-scenes", type=int, default=0,
                    help="only the first N scenes (smoke test)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for exe in ("ffmpeg", "ffprobe"):
        if shutil.which(exe) is None:
            raise SystemExit(
                f"{exe} not found; on ACES run `module load FFmpeg/7.0.2`")
    print(C.describe_paths())

    scenes: dict[tuple[str, str], int] = {}
    for t in [x.strip() for x in args.tasks.split(",") if x.strip()]:
        for r in C.load_manifest(t):
            key = (r["dataset"], r["scene"])
            scenes[key] = scenes.get(key, 0) + 1
    items = sorted(scenes.items(), key=lambda kv: -kv[1])
    if args.limit_scenes:
        items = items[:args.limit_scenes]

    # A crashed run can leave a per-scene lock behind. Extraction holds a lock
    # for seconds, so anything older than 30 minutes is orphaned.
    if C.FRAME_DIR.exists():
        import time
        now = time.time()
        orphaned = [d for d in C.FRAME_DIR.rglob(".extract.lock")
                    if now - d.stat().st_mtime > 1800]
        for d in orphaned:
            d.rmdir()
        if orphaned:
            print(f"cleared {len(orphaned)} orphaned .extract.lock dirs")

    print(f"{len(items)} unique scenes, {sum(n for _, n in items)} questions, "
          f"{C.N_FRAMES} frames each -> {C.FRAME_DIR}")

    done = skipped = failed = 0
    for (ds, scene), nq in items:
        outdir = C.scene_frames_dir(ds, scene)
        marker = outdir / "frames.json"
        if marker.exists() and not args.force:
            meta = json.loads(marker.read_text())
            n_jpg = len(list(outdir.glob("*.jpg")))
            if meta.get("n_extracted") == C.N_FRAMES and n_jpg == C.N_FRAMES:
                skipped += 1
                continue
        src = C.video_path(ds, scene)
        if not src.exists():
            print(f"  [miss] {src}")
            failed += 1
            continue
        try:
            meta = extract(src, outdir, C.N_FRAMES, C.FRAME_MAX_SIDE)
        except Exception as e:                      # noqa: BLE001
            print(f"  [fail] {ds}/{scene}: {e}")
            failed += 1
            continue
        done += 1
        flag = "" if meta["n_extracted"] == C.N_FRAMES else "  <-- SHORT"
        if meta["n_extracted"] != C.N_FRAMES:
            failed += 1
        print(f"  [{done}/{len(items)}] {ds}/{scene} "
              f"src={meta['source_frames']} frames, {nq} questions{flag}")

    print(f"\nextracted {done}, already cached {skipped}, failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
