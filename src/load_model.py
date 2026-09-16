"""Load a model + tokenizer from a declarative config entry.

Every quantization scheme (fp16 / bnb-nf4 / awq / gptq) goes through this
one function so that the only thing that can differ between an fp16 run
and a quantized run is what happens in here -- generation config is
handled entirely separately (see configs/models.yaml `generation` block
and run_eval.py) and must stay identical across all configs.

No hardcoded paths: model IDs and cache locations come from config /
HF's own env vars (HF_HOME, HF_TOKEN), never from anything local to a
particular machine.
"""

import os

# Must be set before any CUDA context is created (i.e. before the first
# CUDA op runs), or torch.use_deterministic_algorithms(True) below will
# raise for cuBLAS ops like matmul instead of running deterministically.
# Setting it here, at import time, guarantees that regardless of which
# entry point (run_eval.py, check_determinism.py, run_milestone.py)
# imports this module first. setdefault so an operator's own env setting
# is respected if already present.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def set_deterministic_seed(seed: int, strict: bool = True) -> None:
    """Seed every RNG the pipeline touches. Call this before load_model
    and again immediately before generation -- lm-eval-harness also
    consumes randomness (few-shot sampling), so both call sites matter.

    strict=True (default) makes torch raise on any op without a
    deterministic CUDA implementation, rather than silently falling back
    to a nondeterministic one with only a warning. That's the correct
    default for this project: a quantization backend whose kernels can't
    run deterministically is a real finding to record, not something to
    paper over by quietly downgrading to warn_only.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def _dtype(name: str) -> torch.dtype:
    if name not in _DTYPES:
        raise ValueError(f"Unknown dtype '{name}', expected one of {list(_DTYPES)}")
    return _DTYPES[name]


def _build_quantization_config(quant: dict | None) -> BitsAndBytesConfig | None:
    """Only bitsandbytes needs a quantization_config object passed to
    from_pretrained. AWQ and GPTQ checkpoints carry their quantization
    metadata in the checkpoint's own config.json and are loaded like any
    other pretrained model once autoawq / auto-gptq are installed --
    transformers dispatches to the right backend automatically.
    """
    if quant is None:
        return None

    method = quant.get("method")
    if method == "bitsandbytes":
        return BitsAndBytesConfig(
            load_in_4bit=quant.get("load_in_4bit", True),
            bnb_4bit_quant_type=quant.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=_dtype(quant.get("bnb_4bit_compute_dtype", "float16")),
            bnb_4bit_use_double_quant=quant.get("bnb_4bit_use_double_quant", True),
        )
    if method in ("awq", "gptq"):
        # Pre-quantized checkpoint; nothing to build here.
        return None

    raise ValueError(f"Unknown quantization method '{method}'")


def apply_generation_config(model, generation_cfg: dict) -> None:
    """Overwrite whatever generation_config.json shipped with the
    checkpoint with the shared block from configs/models.yaml.

    from_pretrained loads each checkpoint's own generation_config.json,
    and Qwen2.5-Instruct checkpoints ship sampling defaults (do_sample:
    true, temperature: 0.7, top_p: 0.8, top_k: 20) baked in. Left alone,
    those silently win over the "greedy, deterministic" config this
    project requires, and the base/AWQ/bnb repos -- separate uploads --
    have no guarantee of shipping identical defaults either. This makes
    the YAML block authoritative instead of decorative.
    """
    model.generation_config.do_sample = generation_cfg["do_sample"]
    model.generation_config.temperature = generation_cfg["temperature"]
    model.generation_config.top_p = generation_cfg["top_p"]
    model.generation_config.top_k = generation_cfg["top_k"]
    model.generation_config.max_new_tokens = generation_cfg["max_new_tokens"]


def load_model(model_cfg: dict, seed: int, generation_cfg: dict):
    """model_cfg is one entry from configs/models.yaml `models:`.
    generation_cfg is the shared `generation:` block, applied here so it
    can't be silently shadowed by a checkpoint's own generation_config.json.
    """
    set_deterministic_seed(seed)

    base_model = model_cfg["base_model"]
    dtype = _dtype(model_cfg.get("dtype", "float16"))
    quantization_config = _build_quantization_config(model_cfg.get("quantization"))

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    # Force pad token/side explicitly rather than trusting each
    # checkpoint repo's shipped tokenizer_config.json to agree -- the
    # base and AWQ repos are separate uploads and there is no guarantee
    # they set these identically, and a divergence here would silently
    # break the "identical tokenization across configs" constraint.
    # Left padding is required for correct batched causal-LM generation
    # (all sequences must end at the same position); pinning it now also
    # means nothing has to change when batch_size later grows past 1.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model.eval()
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    apply_generation_config(model, generation_cfg)
    return model, tokenizer
