import time
import torch
import weakref
import comfy.utils
import comfy.lora
import folder_paths


# ----------------------------------------------------------------------
# LoRA Scheduled (timestep) — DiT-only, chainable, stack-proof.
# Three axes of control:
#   WHEN  — inject_at / stop_at / fade   (window over denoise progress)
#   WHERE — apply_to: both / positive / negative   (CFG branch routing)
#   HOW   — lora_strength
#
# apply_to routes the LoRA into only one side of the CFG split:
#   both      — normal, affects cond and uncond (like a stock loader)
#   positive  — LoRA only in the cond pass; pushes the image toward the
#               LoRA, amplified cleanly through CFG, uncond stays neutral
#   negative  — LoRA only in the uncond pass; the model is pushed AWAY
#               from the LoRA (use any LoRA as a negative concept)
# Requires CFG > 1 to have a separate uncond pass; at CFG=1 positive/
# negative behave like 'both' (only one pass exists).
#
# Key mapping comes from comfy.lora.model_lora_keys_unet, covering every
# layer (including underscore names like cross_attn / q_proj), exactly
# like the stock loader. up/down/alpha are read from the file directly.
# Computed in fp32. Survives Anima dynamic VRAM loading (adds to output).
# ----------------------------------------------------------------------

_MOD_REG = weakref.WeakKeyDictionary()
_ALL_ENTRIES = {}
_ACTIVE_KEYS = set()
_BATCH_INFO = {"cou": None}
_DBG = {"n": 0}


def _get_submodule(root, dotted):
    cur = root
    for part in dotted.split('.'):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            if not hasattr(cur, part):
                return None
            cur = getattr(cur, part)
    return cur


def _parse_lora_with_map(sd, key_map):
    groups = {}

    def add(base, kind, v):
        groups.setdefault(base, {})[kind] = v

    for k, v in sd.items():
        if k.endswith(".lora_down.weight"):
            add(k[:-len(".lora_down.weight")], "down", v)
        elif k.endswith(".lora_up.weight"):
            add(k[:-len(".lora_up.weight")], "up", v)
        elif k.endswith(".lora_A.weight"):
            add(k[:-len(".lora_A.weight")], "down", v)
        elif k.endswith(".lora_B.weight"):
            add(k[:-len(".lora_B.weight")], "up", v)
        elif k.endswith(".alpha"):
            add(k[:-len(".alpha")], "alpha", v)

    result = {}
    unmapped = []
    for base, d in groups.items():
        if "up" not in d or "down" not in d:
            continue
        model_key = key_map.get(base)
        if model_key is None:
            unmapped.append(base)
            continue
        up = d["up"].float()
        down = d["down"].float()
        if up.ndim != 2 or down.ndim != 2:
            unmapped.append(base)
            continue
        rank = down.shape[0]
        alpha = float(d["alpha"]) if "alpha" in d else float(rank)
        result[model_key] = (up, down, alpha / rank)
    return result, unmapped


def _branch_mask(batch, cou, want_cond, want_uncond, device, dtype):
    # cou is the cond_or_uncond list: 0 = cond (positive), 1 = uncond.
    # The batch is split into len(cou) equal chunks along dim 0.
    if cou is None or len(cou) == 0:
        return None
    n = len(cou)
    if batch % n != 0:
        return None
    chunk = batch // n
    mask = torch.zeros(batch, device=device, dtype=dtype)
    for i, val in enumerate(cou):
        lo = i * chunk
        hi = lo + chunk
        is_cond = (int(val) == 0)
        if (is_cond and want_cond) or ((not is_cond) and want_uncond):
            mask[lo:hi] = 1.0
    return mask


def _ensure_hook(module):
    if getattr(module, "_sched_lora_installed", False):
        return
    orig_forward = module.forward

    def fwd(x, *a, **kw):
        out = orig_forward(x, *a, **kw)
        contribs = _MOD_REG.get(module)
        if not contribs or not _ACTIVE_KEYS:
            return out
        add_total = None
        for key, c in contribs.items():
            if key not in _ACTIVE_KEYS:
                continue
            holder = c["holder"]
            w = holder.get("w", 0.0)
            if abs(w) < 1e-6:
                continue
            dev = x.device
            cache = c["cache"].get(dev)
            if cache is None:
                cache = (c["up"].to(dev), c["down"].to(dev))
                c["cache"][dev] = cache
            up_dev, down_dev = cache
            xin = x.float()
            add = torch.nn.functional.linear(
                torch.nn.functional.linear(xin, down_dev), up_dev)
            add = add * (c["scale"] * w)

            apply_to = holder.get("apply_to", "both")
            if apply_to != "both":
                want_cond = (apply_to == "positive")
                want_uncond = (apply_to == "negative")
                mask = _branch_mask(
                    add.shape[0], _BATCH_INFO.get("cou"),
                    want_cond, want_uncond, add.device, add.dtype)
                if mask is not None:
                    shape = [add.shape[0]] + [1] * (add.ndim - 1)
                    add = add * mask.view(shape)

            add_total = add if add_total is None else add_total + add
        if add_total is None:
            return out
        return out + add_total.to(out.dtype)

    module.forward = fwd
    module._sched_lora_installed = True


def _smooth(f):
    f = max(0.0, min(1.0, f))
    return f * f * (3.0 - 2.0 * f)


def _make_wrapper(my_keys):
    def wrapper(apply_model, args):
        t = args["timestep"]
        try:
            cur_sigma = float(t.flatten()[0])
        except Exception:
            cur_sigma = 0.0

        c = args.get("c", {})
        topts = c.get("transformer_options", {}) if isinstance(c, dict) else {}
        _BATCH_INFO["cou"] = topts.get("cond_or_uncond")

        for key in my_keys:
            e = _ALL_ENTRIES.get(key)
            if e is None:
                continue
            s_a = e["s_a"]
            s_b = e["s_b"]
            s_c = e["s_c"]
            s_d = e["s_d"]
            strength = e["strength"]

            if cur_sigma > s_a or cur_sigma < s_d:
                w = 0.0
            elif cur_sigma >= s_b:
                frac = (s_a - cur_sigma) / max(1e-6, (s_a - s_b))
                w = strength * _smooth(frac)
            elif cur_sigma > s_c:
                w = strength
            else:
                frac = (cur_sigma - s_d) / max(1e-6, (s_c - s_d))
                w = strength * _smooth(frac)
            e["holder"]["w"] = w

        if _DBG["n"] < 10:
            print(f"[LoRAScheduled] sigma={cur_sigma:.3f} cou={_BATCH_INFO['cou']} "
                  f"active={list(my_keys)} "
                  f"w={ {k: round(_ALL_ENTRIES[k]['holder']['w'],3) for k in my_keys if k in _ALL_ENTRIES} }")
            _DBG["n"] += 1

        prev = set(_ACTIVE_KEYS)
        _ACTIVE_KEYS.clear()
        _ACTIVE_KEYS.update(my_keys)
        try:
            return apply_model(args["input"], t, **args["c"])
        finally:
            _ACTIVE_KEYS.clear()
            _ACTIVE_KEYS.update(prev)
    return wrapper


class LoRAScheduledTimestep:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "enabled": ("BOOLEAN", {"default": True}),
                "apply_to": (["both", "positive", "negative"], {"default": "both"}),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -3.0, "max": 3.0, "step": 0.05}),
                "inject_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "stop_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                "force_rerun": ("BOOLEAN", {"default": True}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, model, lora_name, enabled, apply_to, lora_strength,
                   inject_at, stop_at, fade, force_rerun, unique_id=None):
        base = (f"{unique_id}|{lora_name}|{enabled}|{apply_to}|{lora_strength}"
                f"|{inject_at}|{stop_at}|{fade}")
        if force_rerun:
            return base + f"|{time.time()}"
        return base

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/lora_schedule"

    def apply(self, model, lora_name, enabled, apply_to, lora_strength,
              inject_at, stop_at, fade, force_rerun, unique_id=None):
        m = model.clone()

        if not enabled:
            return (m,)

        if stop_at <= inject_at:
            stop_at = min(1.0, inject_at + 0.01)

        path = folder_paths.get_full_path("loras", lora_name)
        sd = comfy.utils.load_torch_file(path, safe_load=True)

        key_map = comfy.lora.model_lora_keys_unet(m.model, {})
        deltas, unmapped = _parse_lora_with_map(sd, key_map)

        dm = m.model.diffusion_model

        ms = m.model.model_sampling
        mid = (inject_at + stop_at) / 2.0
        p_a = inject_at
        p_b = min(mid, inject_at + fade)
        p_c = max(mid, stop_at - fade)
        p_d = stop_at
        s_a = float(ms.percent_to_sigma(float(p_a)))
        s_b = float(ms.percent_to_sigma(float(p_b)))
        s_c = float(ms.percent_to_sigma(float(p_c)))
        s_d = float(ms.percent_to_sigma(float(p_d)))

        key = f"{lora_name}#{unique_id}"
        holder = {"w": 0.0, "apply_to": apply_to}
        _ALL_ENTRIES[key] = {
            "holder": holder, "s_a": s_a, "s_b": s_b,
            "s_c": s_c, "s_d": s_d, "strength": float(lora_strength),
        }

        _DBG["n"] = 0
        matched = 0
        skipped = []
        for model_key, (up, down, scale) in deltas.items():
            if model_key.startswith("diffusion_model.") and model_key.endswith(".weight"):
                sub = model_key[len("diffusion_model."):-len(".weight")]
            elif model_key.endswith(".weight"):
                sub = model_key[:-len(".weight")]
            else:
                sub = model_key
            mod = _get_submodule(dm, sub)
            if mod is None or not hasattr(mod, "weight"):
                skipped.append(model_key)
                continue
            _ensure_hook(mod)
            reg = _MOD_REG.get(mod)
            if reg is None:
                reg = {}
                _MOD_REG[mod] = reg
            reg[key] = {
                "up": up.float(), "down": down.float(),
                "scale": scale, "holder": holder, "cache": {},
            }
            matched += 1

        print(f"[LoRAScheduled] {key}: matched={matched} "
              f"skipped={len(skipped)} unmapped={len(unmapped)} "
              f"apply_to={apply_to} window[{inject_at}..{stop_at}] "
              f"fade={fade} strength={lora_strength}")
        if unmapped:
            print(f"[LoRAScheduled] WARNING: {len(unmapped)} unmapped keys "
                  f"(LoRA may not apply fully). First: {unmapped[0]}")
        if skipped:
            print(f"[LoRAScheduled] WARNING: {len(skipped)} mapped keys had no "
                  f"matching module. First: {skipped[0]}")

        keys = list(m.model_options.get("_sched_keys", []))
        if key not in keys:
            keys.append(key)
        m.model_options["_sched_keys"] = keys

        m.set_model_unet_function_wrapper(_make_wrapper(keys))
        return (m,)


NODE_CLASS_MAPPINGS = {"LoRAScheduledTimestep": LoRAScheduledTimestep}
NODE_DISPLAY_NAME_MAPPINGS = {"LoRAScheduledTimestep": "LoRA Scheduled (timestep)"}
