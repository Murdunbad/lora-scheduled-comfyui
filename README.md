# LoRA Scheduled (Timestep)

A custom ComfyUI node that applies a LoRA only during a **specific window of the denoising process** — and, optionally, only to **one side of the CFG split** — instead of throughout the entire generation.

This lets you keep a base model's compositional diversity (poses, camera angles, framing) while still injecting a LoRA's style or character at the right moment. It was developed and tested with **Anima / Qwen-Image** DiT models, but works with any LoRA whose keys map onto the diffusion model.

---

## Why?

Applying a LoRA at full strength for the whole generation forces the model toward the LoRA's learned distribution from the very first step — which locks in composition and kills variety across seeds.

But the **composition of an image is decided in the early denoising steps**, while **style, character identity, and fine detail emerge later**. By switching a LoRA on *after* the composition has formed, you get:

- **Diverse poses and camera angles** (driven by the base model)
- **Consistent character / style** (driven by the LoRA, applied late)

---

## How it works

Unlike a normal LoRA loader that merges weights, this node **hooks the forward pass** of the affected modules and adds the LoRA's low-rank contribution to their output, scaled by a weight that depends on the current denoising progress. The contribution is computed in fp32 for accuracy, and can be routed to only the positive or negative CFG branch (see below).

---

## Installation

### Option 1 — ComfyUI Manager (Install via Git URL)

If you have [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) installed:

1. Open **ComfyUI Manager**.
2. Click **Install via Git URL**.
3. Paste this repository's URL:
~~~
https://github.com/Murdunbad/lora-scheduled-comfyui
~~~
4. Click **OK**, then **Restart** ComfyUI when prompted.

### Option 2 — Manual

1. Clone or download this repository into your ComfyUI `custom_nodes` folder:
~~~
ComfyUI/custom_nodes/lora_scheduled/
~~~
   so that `__init__.py` sits inside that folder.
2. Restart ComfyUI.

After installation the node appears under **`advanced/lora_schedule → LoRA Scheduled (timestep)`**.

---

## Parameters

| Parameter | Description |
|---|---|
| `model` | Model input. Chainable — connect the output to another node of this type to stack multiple scheduled LoRAs. |
| `lora_name` | The LoRA file to load (from your `models/loras` folder). |
| `enabled` | Toggle the LoRA on/off. Unlike bypass, the node still executes when disabled, so `force_rerun` keeps working. |
| `apply_to` | Which CFG branch the LoRA affects: `both`, `positive` (cond only), or `negative` (uncond only). See [Targeting the CFG branch](#targeting-the-cfg-branch). |
| `lora_strength` | LoRA strength, same meaning as in a normal loader. Higher = stronger effect. |
| `inject_at` | Denoise percent at which the LoRA **turns on** (`0.0` = start, `1.0` = end). |
| `stop_at` | Denoise percent at which the LoRA **turns off**. |
| `fade` | Smoothing applied to both edges of the window (`0.0` = hard on/off, `0.1`–`0.2` = soft). |
| `force_rerun` | `True`: every queue forces a fresh generation (useful while tuning, prevents cached results). `False`: normal ComfyUI caching. |

> **Progress convention:** `0.0` is the first denoising step (pure noise), `1.0` is the final step (finished image). The LoRA is active **between** `inject_at` and `stop_at`.

---

## Targeting the CFG branch (`apply_to`)

When CFG > 1, the model runs two predictions per step: a **positive (cond)** pass using your prompt, and a **negative (uncond)** pass. The final result is `uncond + cfg · (cond − uncond)`. A normal LoRA loader edits the weights, so it affects **both** passes equally. This node can route the LoRA into only one of them:

| Mode | Effect |
|---|---|
| `both` | Default. Affects cond and uncond, like a stock loader. |
| `positive` | LoRA applied **only to the cond pass**. Pushes the image *toward* the LoRA, amplified cleanly through CFG while the uncond pass stays a neutral reference. Tends to look cleaner / more saturated, with fewer artifacts than raising global strength. |
| `negative` | LoRA applied **only to the uncond pass**. The model is pushed *away* from the LoRA — effectively turning any LoRA into a negative concept (e.g. push away from a style or unwanted trait). |

**Tips:**

- `positive` is the most generally useful mode — it gives you a cleaner version of "more LoRA" without the pile-up you get from cranking `lora_strength` globally.
- `negative` is a "push-away" tool. Its effect is by nature more diffuse than `positive` (the model can move in many directions to *avoid* the LoRA), so use a **higher strength (2.0–3.0)** and a LoRA with a distinctive style to see a clear result.
- **Requires CFG > 1.** At CFG = 1 (distilled/cfg-free sampling) there is no separate uncond pass: `positive` behaves like `both`, and `negative` has nothing to apply to.

---

## Recommended settings (feel free to adjust and find optimal)

Let the base model build the composition, then bring the LoRA in:

~~~
apply_to      = positive
lora_strength = 1.0
inject_at     = 0.10 – 0.20 (start with 0.20)
stop_at       = 1.0
fade          = 0.0 - 0.05
~~~

- **`inject_at` is your main dial.** Lower (0.1) = more fidelity, the LoRA locks in sooner. Higher (0.3) = more pose/angle diversity, the effect is slightly softer.
- Values above ~0.3 tend to weaken the result noticeably; values below ~0.1 start to constrain composition like a normal loader. **0.10–0.20 is the sweet spot.**
- Because the LoRA is off during composition, you have more headroom on `lora_strength` than you normally would.
- Try `apply_to = positive` together with a late `inject_at`: composition stays free (base model), and the LoRA arrives cleanly through CFG without polluting the uncond reference.

> **Base model matters.** This node relies on the base model's own compositional variety during the early steps. **Merged checkpoints tend to have significantly reduced diversity** — their compositions are already collapsed toward a narrow distribution, so there is little variety left for the node to preserve. For best results use a clean base model rather than a merge; on merges the benefit of timestep scheduling is largely lost.

---

## Chaining multiple LoRAs

The node is chainable. Each instance manages its own window **and its own `apply_to` branch**, so you can combine LoRAs that operate in different phases or on different CFG sides:

~~~
base model
  └─ LoRA Scheduled (LoRA A, apply_to positive, inject_at 0.1, stop_at 1.0)
       └─ LoRA Scheduled (LoRA B, apply_to negative, inject_at 0.2, stop_at 1.0)
            └─ KSampler  (single sampler — no split needed)
~~~

Each LoRA can be injected at its own point in the denoise and routed to its own branch, all feeding a single sampler.

---

## Notes & limitations

- **Use a clean base model, not a merge.** Diversity on merged checkpoints is significantly reduced, which defeats the purpose of this node — there is little compositional variety left to preserve. Merges are not recommended.
- If you still want to use a merge, then reduce the number of quality tags and lower the CFG / increase the shift.
- **`apply_to` needs CFG > 1.** With cfg-free / distilled sampling there is no separate uncond pass, so `positive`/`negative` routing has no effect (falls back to `both`).
- **DiT / `diffusion_model.`-namespace LoRAs.** Native-format LoRAs (keys starting with `diffusion_model.`) are fully supported. Kohya-style `lora_unet_` keys are handled on a best-effort basis and may not map perfectly on all architectures.
- **No text-encoder scheduling.** This node schedules the diffusion model (UNet/DiT) portion only.
- The node monkey-patches module `forward` methods on the live model. This is gated and inert when no scheduled LoRA is active, but it is the trade-off that makes dynamic VRAM loading compatibility possible.
- Developed against ComfyUI with Anima / Qwen-Image. Other model families should work where the LoRA keys resolve onto submodules, but they are untested.

---

## License

MIT.
