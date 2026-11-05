"""Qwen3-4B (model-family check, paper appendix). 36 decoder layers."""

from .base import ModelSpec

SPEC = ModelSpec(
    hf_name="Qwen/Qwen3-4B",
    thought_open="<think>\n",
    thought_close="\n</think>",
    answer_sep="\n\n",
    end_of_turn="<|im_end|>",
    layers_path="model.layers",
    steer_layers=(21, 32),
    sft_lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    rl_lora_targets=["q_proj", "o_proj"],
    think_kwargs={"enable_thinking": True},
    plain_kwargs={"enable_thinking": False},
)
