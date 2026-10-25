"""Map the WebUI parameter dict onto a KohakuMoss train config (scripts/train/moss.py).

`build_config(params, steps_per_epoch)` returns the config text plus the derived values
shown in the UI; `to_linux_path` maps Windows paths onto their WSL mounts.
"""

import re

from webui_app.schema import defaults

TOKEN_FACTOR = 16  # pixels per latent token side (VAE /8 x 2x2 patchify)
TEXT_MARGIN = 1024  # text tokens reserved per sample when MAX_TOKENS is derived


def to_linux_path(p) -> str:
    """F:\\a\\b or F:/a/b -> /mnt/f/a/b; other paths are only slash-normalised."""
    if p is None:
        return ""
    p = str(p).strip().strip('"').strip("'")
    if not p:
        return ""
    if len(p) >= 2 and p[1] == ":":
        rest = p[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{p[0].lower()}/{rest}"
    return p.replace("\\", "/")


def safe_run_name(name: str) -> str:
    """Sanitize run name for filesystem paths and file prefixes."""
    return "".join(c for c in str(name or "") if c.isalnum() or c in "-_.").strip() or "run"


def merged(params: dict) -> dict:
    """User params over the schema defaults."""
    out = defaults()
    out.update({k: v for k, v in (params or {}).items() if v is not None})
    return out


def _f(p, key, fallback=0.0) -> float:
    try:
        return float(p[key])
    except (TypeError, ValueError, KeyError):
        return float(fallback)


def _i(p, key, fallback=0) -> int:
    return int(round(_f(p, key, fallback)))


def geometry(p: dict) -> dict:
    """Target area, bucket step, aspect range and token numbers from the resolution knobs."""
    res = max(64, _i(p, "resolution", 1024))
    area = res * res
    step = max(TOKEN_FACTOR, _i(p, "bucket_reso_steps", 32) // TOKEN_FACTOR * TOKEN_FACTOR)
    lo_side = max(TOKEN_FACTOR, _i(p, "min_bucket_reso", 256))
    hi_side = max(lo_side, _i(p, "max_bucket_reso", 3072))
    ratio = min(hi_side**2 / area, area / lo_side**2)
    ratio = max(1.0, ratio)
    img_tokens = (res // TOKEN_FACTOR) ** 2
    return {
        "resolution": res,
        "target_area": area,
        "step": step,
        "aspect_range": (round(1.0 / ratio, 6), round(ratio, 6)),
        "min_side": lo_side,
        "max_side": hi_side,
        "max_image_tokens": img_tokens,
    }


def dataset_entries(p: dict) -> list[dict]:
    """DATASETS knob entries from the dataset list parameter (WSL paths)."""
    txt_as = "nl" if p["caption_template"] == "nl" else "tags"
    entries = []
    for item in p.get("datasets") or []:
        path = to_linux_path(item.get("path"))
        if not path:
            continue
        entry = {"source": item.get("source") or "kohya", "path": path}
        if entry["source"] == "kohya":
            entry["repeat_prefix"] = "first"
            entry["txt_as"] = txt_as
            if p.get("artist_from_folder"):
                entry["artist_from_folder"] = "name"
            if item.get("include_root"):
                entry["include_root"] = True
        else:
            entry["txt_as"] = txt_as
        weight = int(round(float(item.get("weight") or 1.0)))
        if weight != 1:
            entry["weight"] = weight
        entries.append(entry)
    return entries


def caption_knob(p: dict) -> dict:
    keep = ["artist"] if p.get("artist_from_folder") else []
    if p["caption_template"] == "nl":
        return {"template": "nl", "keep": keep}
    return {
        "template": "tags",
        "keep": keep,
        "keep_first": _i(p, "keep_tokens"),
        "shuffle": bool(p["shuffle_caption"]),
        "tag_dropout": _f(p, "tag_dropout"),
        "underscores": bool(p["underscores"]),
        "unescape": True,
        "prefix": str(p.get("caption_prefix") or ""),
        "suffix": str(p.get("caption_suffix") or ""),
    }


def algo_kwargs(p: dict) -> dict:
    match p["algo"]:
        case "lokr":
            return {
                "factor": _i(p, "lokr_factor", 8),
                "decompose_both": bool(p["lokr_decompose_both"]),
                "full_matrix": bool(p["lokr_full_matrix"]),
                "use_tucker": bool(p["use_tucker"]),
                "rank_dropout_scale": bool(p["rank_dropout_scale"]),
            }
        case "lora" | "locon" | "loha":
            return {
                "use_tucker": bool(p["use_tucker"]),
                "use_scalar": bool(p["use_scalar"]),
                "rank_dropout_scale": bool(p["rank_dropout_scale"]),
            }
        case _:
            return {}


def preset_name(p: dict) -> str:
    name = p["preset"]
    for key, suffix in (
        ("preset_text_refiner", "+text-refiner"),
        ("preset_modulation", "+modulation"),
        ("preset_embed", "+embed"),
    ):
        if p.get(key):
            name += suffix
    return name


def schedule_knob(p: dict, steps: int) -> dict:
    warmup = max(0, min(_i(p, "lr_warmup_steps"), steps - 1))
    match p["lr_scheduler"]:
        case "constant":
            return {"mode": "constant", "warmup": warmup, "end": -1}
        case "linear":
            return {
                "mode": "polynomial",
                "power": 1,
                "warmup": warmup,
                "end": -1,
                "min_value": _f(p, "lr_min_ratio"),
            }
        case _:
            return {
                "mode": "cosine",
                "warmup": warmup,
                "end": -1,
                "min_value": _f(p, "lr_min_ratio"),
            }


def seeds_tuple(text) -> tuple:
    out = []
    for part in re.split(r"[,\s]+", str(text or "")):
        if part.strip().lstrip("-").isdigit():
            out.append(int(part))
    return tuple(out) or (0,)


def steps_plan(p: dict, steps_per_epoch: int) -> dict:
    """Total steps / save / sample intervals in optimiser steps."""
    spe = max(1, int(steps_per_epoch))
    if p["duration_unit"] == "epochs":
        steps = max(1, _i(p, "max_epochs", 1) * spe)
        save = max(0, _i(p, "save_every_epochs", 10) * spe)
        export_lora = max(0, _i(p, "save_lora_every_epochs", 1) * spe)
        sample = max(0, _i(p, "sample_every_epochs", 1) * spe)
    else:
        steps = max(1, _i(p, "max_steps", 1000))
        save = max(0, _i(p, "save_every_steps", 500))
        export_lora = max(0, _i(p, "save_lora_every_steps", 100))
        sample = max(0, _i(p, "sample_every_steps", 100))
    return {
        "steps": steps,
        "save_every": save,
        "export_every": export_lora,
        "sample_every": sample,
        "steps_per_epoch": spe,
    }


def fmt(value) -> str:
    return repr(value)


def data_overrides(params: dict) -> str:
    """Knob lines appended to a train config to make it drive the cache scripts: the
    text / image encoder factories and the weight files they read."""
    p = merged(params)
    root = to_linux_path(p["weights_root"])
    text = to_linux_path(p["text_encoder"]) or root
    vae = to_linux_path(p["vae"]) or root
    device = "cpu" if p["text_device"] == "cpu" else "cuda"
    return "\n".join([
        "",
        "# ===== CACHE ENCODERS (data scripts) =====",
        "TEXT_ENCODER = 'kohakumoss.encoders.text_encoder'",
        f"TEXT_ENCODER_KWARGS = {fmt({'path': text, 'device': device})}",
        "IMAGE_ENCODER = 'kohakumoss.encoders.vae_encoder'",
        f"IMAGE_ENCODER_KWARGS = {fmt({'path': vae, 'device': 'cuda'})}",
        "",
    ])


def build_config(params: dict, steps_per_epoch: int = 100, run_dir: str = "") -> tuple[str, dict]:
    """(config.py text, derived dict) for `params` (missing keys take schema defaults)."""
    p = merged(params)
    geo = geometry(p)
    plan = steps_plan(p, steps_per_epoch)
    steps = plan["steps"]
    batch_size = max(1, _i(p, "batch_size", 1))
    grad_acc = max(1, _i(p, "grad_acc", 1))
    batching = p["batching"]
    per_sample = geo["max_image_tokens"] + TEXT_MARGIN
    if batching == "count":
        max_tokens = batch_size * per_sample
    else:
        max_tokens = max(_i(p, "max_tokens", 12288), per_sample)
    bucket = bool(p["bucket_enable"])
    mode = "bucket" if bucket else p["non_bucket_mode"]
    algo = p["algo"]
    lokr = algo == "lokr"
    dim = 10000 if lokr else _i(p, "dim", 32)
    alpha = 1.0 if lokr else _f(p, "alpha", dim)
    out_dir = to_linux_path(p["output_dir"]) or "runs/train"
    lines: list[str] = []
    add = lines.append

    add('"""Generated by the KohakuMoss WebUI; drives scripts/train/moss.py."""')
    add("")
    add("# ===== MODEL =====")
    add(f"WEIGHTS = {fmt(to_linux_path(p['weights_root']))}")
    add(f"VARIANT = {fmt(p['variant'])}")
    add(f"TRANSFORMER = {fmt(to_linux_path(p['transformer']))}")
    add(f"TEXT_ENCODER = {fmt(to_linux_path(p['text_encoder']))}")
    add(f"VAE = {fmt(to_linux_path(p['vae']))}")
    add(f"TEXT_DEVICE = {fmt(p['text_device'])}")
    text_enc = {"memory": p["text_memory"], "ahead": 4}
    if p["text_precision"] != "auto":
        text_enc["precision"] = p["text_precision"]
    add(f"TEXT_ENCODE = {fmt(text_enc)}")
    add(f"VAE_DTYPE = {fmt(p['vae_dtype'])}")
    add(f"VAE_ENCODE = {fmt({'memory': p['vae_memory'], 'tile': bool(p['vae_tile'])})}")
    add(f"TRAIN_TEXT_REFINER = {fmt(bool(p['train_text_refiner']))}")
    add("")
    add("# ===== DATA =====")
    add(f"DATASETS = {fmt(dataset_entries(p))}")
    add(f"RESOLUTION_MODE = {fmt(mode)}")
    add(f"TARGET_AREA = {geo['target_area']}")
    add(f"SIZE_STEP = {geo['step']}")
    add(f"ASPECT_RANGE = {fmt(geo['aspect_range'])}")
    add(f"UPSCALE = {fmt('never' if p['no_upscale'] else 'to_area')}")
    add(f"CROP = {fmt(p['crop'])}")
    add(f"FLIP = {fmt(bool(p['flip']))}")
    add(f"BACKGROUND = {fmt(p['background'])}")
    add(f"RESAMPLE = {fmt(p['resample'])}")
    add(f"BATCHING = {fmt(batching)}")
    add(f"BATCH_SIZE = {batch_size}")
    add(f"MAX_TOKENS = {max_tokens}")
    add(f"OVERLONG = {fmt(p['overlong'])}")
    add(f"PLANNER_KWARGS = {fmt({'drop_last': bool(p['drop_last'])})}")
    add(f"CAPTION = {fmt(caption_knob(p))}")
    add(f"CAPTION_VARIANTS = {_i(p, 'caption_variants', 4)}")
    add(f"TEXT_DROP = {_f(p, 'text_drop', 0.1)}")
    add(f"LATENT_CACHE = {fmt(p['latent_cache'])}")
    add(f"TEXT_CACHE = {fmt(p['text_cache'])}")
    add(f"CACHE_DIR = {fmt(to_linux_path(p['cache_dir']))}")
    add(f"CACHE_DTYPE = {fmt(p['cache_dtype'])}")
    add(f"NUM_WORKERS = {_i(p, 'num_workers', 4)}")
    add(f"PREFETCH = {_i(p, 'prefetch', 2)}")
    add(f"SEED = {_i(p, 'seed', 42)}")
    add("")
    add("# ===== ADAPTER =====")
    add(f"ALGO = {fmt(algo)}")
    add(f"PRESET = {fmt(preset_name(p))}")
    add(f"DIM = {dim}")
    add(f"ALPHA = {alpha}")
    add(f"ALGO_KWARGS = {fmt(algo_kwargs(p))}")
    add(f"DORA = {fmt(bool(p['dora']) and algo in ('lora', 'locon'))}")
    add(f"DORA_DETACH = {fmt(bool(p['dora_detach']))}")
    add(
        "DROPOUT = "
        + fmt({
            "input": _f(p, "dropout_input"),
            "rank": _f(p, "dropout_rank"),
            "module": _f(p, "dropout_module"),
        })
    )
    add(f"ADAPTER_MODE = {fmt(p['adapter_mode'])}")
    add(f"LORAPLUS_RATIO = {_f(p, 'loraplus_ratio', 1.0)}")
    add(f"TRAIN_NORMS = {fmt(bool(p['train_norms']))}")
    add("")
    add("# ===== OBJECTIVE =====")
    add(f"TIMESTEP = {fmt(p['timestep'])}")
    if p["timestep"] == "moss":
        kw = {
            "mu": _f(p, "timestep_mu", -1.09),
            "s": _f(p, "timestep_s", 0.95),
            "shift": _f(p, "timestep_shift", 1.0),
            "cutoff": 0.95,
            "uniform_mix": _f(p, "timestep_uniform_mix", 0.01),
        }
        add(f"TIMESTEP_KWARGS = {fmt(kw)}")
    add(f"LOSS = {fmt(p['loss'])}")
    add(f"WEIGHTING = {fmt(p['weighting'])}")
    add("")
    add("# ===== OPTIM =====")
    add(f"OPTIMIZER = {fmt(p['optimizer'])}")
    add(f"LR = {_f(p, 'learning_rate', 1e-4)!r}")
    add(f"BETAS = {fmt((_f(p, 'beta1', 0.9), _f(p, 'beta2', 0.95)))}")
    add(f"WEIGHT_DECAY = {_f(p, 'weight_decay', 0.01)}")
    add(f"SCHEDULE = {fmt(schedule_knob(p, steps))}")
    add(f"GRAD_CLIP = {_f(p, 'grad_clip', 1.0)}")
    add(f"EMA = {_f(p, 'ema', 0.0)}")
    add("")
    add("# ===== PRECISION & MEMORY =====")
    add(f"PRECISION = {fmt(p['precision'])}")
    if p["precision"] == "fp16":
        add("FP16_FIX = 'online'")
        add("FP16_FIX_KWARGS = {'c': 1 / 16}")
    add(f"BASE_PRECISION = {fmt(p['base_precision'])}")
    add(f"MEMORY_PROFILE = {fmt(p['memory_profile'])}")
    add(f"MEMORY_AUTO_FORCE = {fmt(bool(p['memory_auto_force']))}")
    add(f"STEP_RELEASE = {fmt(p['step_release'])}")
    if p["memory_profile"] == "manual":
        add(f"CHECKPOINT = {fmt(p['checkpoint'])}")
        add(f"STREAM = {fmt(p['stream'])}")
        add(f"STREAM_RESIDENT = {_i(p, 'stream_resident', 4)}")
        add(f"ACT_OFFLOAD = {fmt(bool(p['act_offload']))}")
        add(f"OPTIM_STATE = {fmt(p['optim_state'])}")
        add(f"GRAD_OFFLOAD = {fmt(bool(p['grad_offload']))}")
        add(f"FUSED_STEP = {fmt(bool(p['fused_step']))}")
    add(f"TUNE = {fmt(p['tune'])}")
    add(f"COMPILE = {fmt(p['compile'])}")
    add("")
    add("# ===== RUNTIME =====")
    add(f"STEPS = {steps}")
    add(f"GRAD_ACC = {grad_acc}")
    add("DEVICES = (0,)")
    add("PARALLEL = 'none'")
    add(f"OUTPUT_DIR = {fmt(out_dir)}")
    add(f"RUN_NAME = {fmt(p['run_name'])}")
    add(f"LOGGER = {fmt(p['logger'])}")
    add(f"LOG_EVERY = {_i(p, 'log_every', 1)}")
    add(f"NONFINITE_PATIENCE = {_i(p, 'nonfinite_patience', 8)}")
    add("")
    add("# ===== SAMPLING =====")
    if p["sample_enable"]:
        prompts = [s.strip() for s in str(p["sample_prompts"]).splitlines() if s.strip()]
        add(f"SAMPLE_EVERY = {plan['sample_every']}")
        add(f"SAMPLE_PROMPTS = {fmt(prompts or ['a lighthouse on a rocky coast at sunset'])}")
        add(f"SAMPLE_NEGATIVE = {fmt(str(p['sample_negative']))}")
        add(f"SAMPLE_STEPS = {_i(p, 'sample_steps', 30)}")
        add(f"SAMPLE_CFG = {_f(p, 'sample_cfg', 3.0)}")
        add(f"SAMPLE_SHIFT = {_f(p, 'sample_shift', 5.0)}")
        add(f"SAMPLE_SIZE = {fmt((_i(p, 'sample_height', 1024), _i(p, 'sample_width', 1024)))}")
        add(f"SAMPLE_SEEDS = {fmt(seeds_tuple(p['sample_seeds']))}")
    else:
        add("SAMPLE_EVERY = 0")
    add("")
    add("# ===== CHECKPOINT / EXPORT =====")
    add(f"SAVE_EVERY = {plan['save_every']}")
    add(f"EXPORT_EVERY = {plan['export_every']}")
    add(f"KEEP_LAST = {_i(p, 'keep_last', 0)}")
    add(f"SAVE_FORMAT = {fmt(p['save_format'])}")
    add(f"SAVE_MERGED = {fmt(bool(p['save_merged']))}")
    if str(p.get("resume") or "").strip():
        resume = str(p["resume"]).strip()
        add(f"RESUME = {fmt('latest' if resume == 'latest' else to_linux_path(resume))}")
    extra = str(p.get("extra_config") or "").strip()
    if extra:
        add("")
        add("# ===== CUSTOM OVERRIDES =====")
        add(extra)
    add("")

    run_dir = f"{out_dir.rstrip('/')}/{p['run_name']}"
    derived = {
        "steps": steps,
        "save_every": plan["save_every"],
        "export_every": plan["export_every"],
        "sample_every": plan["sample_every"] if p["sample_enable"] else 0,
        "steps_per_epoch": plan["steps_per_epoch"],
        "grad_acc": grad_acc,
        "batch_size": batch_size,
        "effective_batch": batch_size * grad_acc,
        "max_tokens": max_tokens,
        "resolution_mode": mode,
        "target_area": geo["target_area"],
        "aspect_range": geo["aspect_range"],
        "size_step": geo["step"],
        "min_side": geo["min_side"],
        "max_side": geo["max_side"],
        "run_dir": run_dir,
    }
    return "\n".join(lines), derived
