#!/usr/bin/env python
"""
Wrapper around the official Bench2DriveZoo `prepare_B2D.py` that builds ONLY the val infos.

Why a wrapper instead of running prepare_B2D.py directly:
  * its __main__ builds the train infos first, and we deliberately downloaded only the 50
    official val clips, so the train pass would iterate an empty list and waste time;
  * its DATAROOT/MAP_ROOT/OUT_DIR are module-level constants written for a different cwd.

The upstream file itself is NOT modified — we import it and override the constants.
Only clips actually present on disk are processed, so this can be run repeatedly while the
remaining clips are still downloading.
"""
import argparse
import json
import multiprocessing
import os
import pickle
import sys
from os.path import join

TOOLS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TOOLS)

# vis_utils.py (vendored from Bench2DriveZoo) does `import open3d as o3d` at module level
# but never calls it -- verified: zero `o3d.*` / `open3d.*` references in the file. Stub it
# so we don't pull a ~400 MB unused dependency into the env. If a future upstream revision
# starts using open3d this stub will raise a clear AttributeError rather than silently
# returning wrong geometry.
if "open3d" not in sys.modules:
    import types

    class _Open3DStub(types.ModuleType):
        def __getattr__(self, name):
            raise AttributeError(
                f"open3d.{name} was accessed, but open3d is stubbed out in "
                "prepare_infos_val.py because vis_utils.py did not use it. Install open3d."
            )

    sys.modules["open3d"] = _Open3DStub("open3d")

import prepare_B2D as P  # noqa: E402  (constants are patched right after import)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", default="/root/VLA/orion/data/bench2drive")
    ap.add_argument("--outdir", default="/root/VLA/orion/data/infos")
    ap.add_argument("--split", default="/root/VLA/orion/data/splits/bench2drive_base_train_val_split.json")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--tmp-dir", default="tmp_data_val")
    ap.add_argument("--skip-map", action="store_true")
    args = ap.parse_args()

    P.DATAROOT = args.dataroot
    P.MAP_ROOT = join(args.dataroot, "maps")
    P.OUT_DIR = args.outdir
    os.makedirs(P.OUT_DIR, exist_ok=True)

    with open(args.split) as f:
        val_list = json.load(f)["val"]

    present = [v for v in val_list if os.path.isdir(join(args.dataroot, v, "anno"))]
    missing = [v for v in val_list if v not in present]
    print(f"val clips listed: {len(val_list)}  present on disk: {len(present)}")
    if missing:
        print(f"  NOT yet extracted ({len(missing)}): {[m.split('/')[-1] for m in missing[:5]]} ...")
    if not present:
        raise SystemExit("no extracted clips found; nothing to do")

    workers = max(1, min(args.workers, len(present)))
    P.process_list = []
    print(f"building val infos with {workers} workers ...")
    P.generate_infos(present, workers, "val", args.tmp_dir)

    out = join(P.OUT_DIR, "b2d_infos_val.pkl")
    with open(out, "rb") as f:
        infos = pickle.load(f)
    folders = {i["folder"] for i in infos}
    print(f"wrote {out}: {len(infos)} frames over {len(folders)} clips")

    if not args.skip_map:
        print("building map infos ...")
        P.gengrate_map(P.MAP_ROOT)
        print("wrote", join(P.OUT_DIR, "b2d_map_infos.pkl"))


if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)
    main()
