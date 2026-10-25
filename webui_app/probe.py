"""Dataset probe, run inside the KohakuMoss venv:

    python -m webui_app.probe <config.py>

Builds the real MossDataset from a generated config (caches off), then prints one line
`@@RESULT@@ {json}` with the catalog statistics, the epoch-0 plan (batches, sizes,
upscaled share, crop loss) and how many images fall outside the bucket side limits.
"""

import collections
import json
import sys

import numpy as np

from kohakumoss_data.dataset import MossDataset
from kohakumoss_data.resolution.geometry import Geometry
from kohakumoss_data.scripting import data_config
from kohakumoss_data.stats import catalog_stats, plan_stats

MARK = "@@RESULT@@"


def load_knobs(path: str) -> dict:
    ns: dict = {}
    with open(path, encoding="utf-8") as f:
        exec(compile(f.read(), path, "exec"), ns)
    return {k: v for k, v in ns.items() if k.isupper()}


def side_report(dataset, min_side: int, max_side: int) -> dict:
    cat = dataset.catalog
    live = np.asarray(cat.live, dtype=bool)
    w, h = np.asarray(cat.w)[live], np.asarray(cat.h)[live]
    short, long_ = np.minimum(w, h), np.maximum(w, h)
    return {
        "below_min_side": int((short < min_side).sum()),
        "above_max_side": int((long_ > max_side).sum()),
        "smallest": [int(short.min()), int(long_[short.argmin()])] if len(w) else [],
        "largest": [int(w[long_.argmax()]), int(h[long_.argmax()])] if len(w) else [],
    }


def main() -> None:
    knobs = load_knobs(sys.argv[1])
    cfg = data_config(knobs, latent_cache="off", text_cache="off")
    dataset = MossDataset(cfg)
    budget = None if cfg.batching == "count" else cfg.max_tokens
    plan = plan_stats(dataset, 0, budget)
    grad_acc = max(1, int(knobs.get("GRAD_ACC", 1)))
    sizes = collections.Counter(dataset.planner.targets)
    geometry = Geometry(
        area=cfg.target_area,
        step=cfg.size_step,
        aspect_range=tuple(cfg.aspect_range),
        upscale=cfg.upscale,
        crop=cfg.crop,
        latent_factor=cfg.latent_factor,
    )
    buckets = geometry.buckets() if cfg.resolution_mode == "bucket" else []
    min_side = min((min(b) for b in buckets), default=0)
    max_side = max((max(b) for b in buckets), default=0)
    result = {
        "catalog": catalog_stats(dataset.catalog),
        "plan": plan,
        "batches": plan["batches"],
        "grad_acc": grad_acc,
        "steps_per_epoch": max(1, plan["batches"] // grad_acc),
        "targets": [[f"{hh}x{ww}", n] for (hh, ww), n in sizes.most_common(40)],
        "distinct_targets": len(sizes),
        "buckets": len(buckets),
        "bucket_min_side": int(min_side),
        "bucket_max_side": int(max_side),
        "sides": side_report(
            dataset, int(knobs.get("MIN_SIDE", 0) or 0), int(knobs.get("MAX_SIDE", 1 << 30))
        ),
    }
    print(MARK + " " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
