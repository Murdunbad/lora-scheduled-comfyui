import time
import torch
import weakref
import comfy.utils
import folder_paths

_MOD_REG = weakref.WeakKeyDictionary()
_ALL_ENTRIES = {}
_ACTIVE_KEYS = set()
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


def _parse_lora(sd):
    groups = {}
    def base_of(k, suf): return k[:-len(suf)]
    for k, v in sd.items():
        if k.endswith(".lora_down.weight"):
            groups.setdefault(base_of(k, ".lora_down.weight"), {})["down"] = v
        elif k.endswith(".lora_up.weight"):
            groups.setdefault(base_of(k, ".lora_up.weight"), {})["up"] = v
        elif k.endswith(".lora_A.weight"):
            groups.setdefault(base_of(k, ".lora_A.weight"), {})["down"] = v
        elif k.endswith(".lora_B.weight"):
            groups.setdefault(base_of(k, ".lora_B.weight"), {})["up"] = v
        elif k.endswith(".alpha"):
            groups.setdefault(base_of(k, ".alpha"), {})["alpha"] = v
    result = {}
    for base, d in groups.items():
        if "up" not in d or "down" not in d:
            continue
        if base.startswith("diffusion_model."):
            path = base[len("diffusion_model."):]
        elif base.startswith("lora_unet_"):
            path = base[len("lora_unet_"):].replace("_", ".")
        else:
            path = base
        up = d["up"].float()
        down = d["down"].float()
        rank = down.shape[0]
        alpha = float(d["alpha"]) if "alpha" in d else float(rank)
        result[path] = (up, down, alpha / rank)
    return result


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
            w = c["holder"].get("w", 0.0)
            if abs(w) < 1e-6:
                continue
            dev = x.device
            cache = c["cache"].get(dev)
            if cache is None:
                cache = (c["up"].to(dev), c["down"].to(dev))
                c["cache"][dev] = cache
            up_dev, down_dev = cache
            xin = x.to(up_dev.dtype)
            add = torch.nn.functional.linear(
                torch.nn.functional.linear(xin, down_dev), up_dev)
            add = add * (c["scale"] * w)
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
            print(f"[LoRAScheduled] sigma={cur_sigma:.3f} active={list(my_keys)} "
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
        return {"required": {
            "model": ("MODEL",),
            "lora_name": (folder_paths.get_filename_list("loras"),),
            "enabled": ("BOOLEAN", {"default": True}),
            "lora_strength": ("FLOAT", {"default": 1.0, "min": -3.0, "max": 3.0, "step": 0.05}),
            "inject_at": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "stop_at": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "fade": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
            "force_rerun": ("BOOLEAN", {"default": True}),
        }}

    @classmethod
    def IS_CHANGED(cls, model, lora_name, enabled, lora_strength,
                   inject_at, stop_at, fade, force_rerun):
        base = f"{lora_name}|{enabled}|{lora_strength}|{inject_at}|{stop_at}|{fade}"
        if force_rerun:
            return base + f"|{time.time()}"
        return base

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/lora_schedule"

    def apply(self, model, lora_name, enabled, lora_strength,
              inject_at, stop_at, fade, force_rerun):
        m = model.clone()

        if not enabled:
            return (m,)

        if stop_at <= inject_at:
            stop_at = min(1.0, inject_at + 0.01)

        path = folder_paths.get_full_path("loras", lora_name)
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        deltas = _parse_lora(sd)
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

        key = lora_name
        holder = {"w": 0.0}
        _ALL_ENTRIES[key] = {
            "holder": holder, "s_a": s_a, "s_b": s_b,
            "s_c": s_c, "s_d": s_d, "strength": float(lora_strength),
        }

        _DBG["n"] = 0
        matched = 0
        for p, (up, down, scale) in deltas.items():
            mod = _get_submodule(dm, p)
            if mod is None or not hasattr(mod, "weight"):
                continue
            _ensure_hook(mod)
            reg = _MOD_REG.get(mod)
            if reg is None:
                reg = {}
                _MOD_REG[mod] = reg
            reg[key] = {
                "up": up.half(), "down": down.half(),
                "scale": scale, "holder": holder, "cache": {},
            }
            matched += 1

        print(f"[LoRAScheduled] {lora_name}: modules={matched} "
              f"window[{inject_at}..{stop_at}] fade={fade} strength={lora_strength} "
              f"(s_a={s_a:.2f} s_b={s_b:.2f} s_c={s_c:.2f} s_d={s_d:.2f})")

        keys = list(m.model_options.get("_sched_keys", []))
        if key not in keys:
            keys.append(key)
        m.model_options["_sched_keys"] = keys

        m.set_model_unet_function_wrapper(_make_wrapper(keys))
        return (m,)


NODE_CLASS_MAPPINGS = {"LoRAScheduledTimestep": LoRAScheduledTimestep}
NODE_DISPLAY_NAME_MAPPINGS = {"LoRAScheduledTimestep": "LoRA Scheduled (timestep)"}
