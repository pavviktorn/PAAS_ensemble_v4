"""transformers 4.37 -> 5.13 compatibility helpers for the CLIP-based detector heads.

Everything in PAAS_ensemble_v4 runs on ONE venv (transformers 5.13 + vLLM). The CLIP detector heads
(MIDS, A1/A2 9-class, GSD, SeLop) were trained on tf4.37.2, whose CLIPVisionModel nested the encoder
under `.vision_model`; tf5.13 flattened it (`.encoder.layers` directly). The CLIP forward is
numerically identical (max abs diff ~9e-4), so the trained weights need only a KEY REMAP -- no
retraining. Also: tf5's T5 tokenizer reports a huge `model_max_length` sentinel; cap it.
"""
T5_MAX_LEN = 512  # tf5 T5Tokenizer.model_max_length is a huge sentinel -> pass a finite cap instead


def remap_clip_state_dict(sd: dict) -> dict:
    """Strip DDP `module.` and the tf4 `...vision_model...` CLIP nesting so tf4-trained CLIP weights
    load into a tf5 CLIPVisionModel (which exposes `.encoder`/`.embeddings`/`.post_layernorm`/
    `.pre_layrnorm` at top level)."""
    out = {}
    for k, v in sd.items():
        k = k.replace("module.", "")
        k = k.replace("vision_model.encoder.", "encoder.")
        k = k.replace("vision_model.embeddings.", "embeddings.")
        k = k.replace("vision_model.pre_layrnorm.", "pre_layrnorm.")
        k = k.replace("vision_model.post_layernorm.", "post_layernorm.")
        k = k.replace("vision_model.", "")          # any remaining nested ref
        out[k] = v
    return out
