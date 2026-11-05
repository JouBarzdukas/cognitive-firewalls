from . import gemma4, glm_z1, qwen3, r1_llama
from .base import ModelSpec, free, load_model, load_tokenizer

MODELS: dict[str, ModelSpec] = {
    "gemma4_e4b": gemma4.SPEC,
    "qwen3_4b": qwen3.SPEC,
    "r1_llama_8b": r1_llama.SPEC,
    "glm_z1_9b": glm_z1.SPEC,
}

__all__ = ["MODELS", "ModelSpec", "free", "load_model", "load_tokenizer"]
