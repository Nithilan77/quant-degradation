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

import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def set_deterministic_seed(seed: int) -> None:
    """Seed every RNG the pipeline touches. Call this before load_model
    and again immediately before generation -- lm-eval-harness also
    consumes randomness (few-shot sampling), so both call sites matter.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


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


def load_model(model_cfg: dict, seed: int):
    """model_cfg is one entry from configs/models.yaml `models:`."""
    set_deterministic_seed(seed)

    base_model = model_cfg["base_model"]
    dtype = _dtype(model_cfg.get("dtype", "float16"))
    quantization_config = _build_quantization_config(model_cfg.get("quantization"))

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model.eval()
    return model, tokenizer
