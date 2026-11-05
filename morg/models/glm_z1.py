"""GLM-Z1-9B-0414 (model-family check, paper appendix). 40 decoder layers."""

from .base import ModelSpec

SPEC = ModelSpec(
    hf_name="zai-org/GLM-Z1-9B-0414",
    thought_open="<think>",
    thought_close="</think>",
    answer_sep="\n",
    end_of_turn="<|user|>",
    layers_path="model.layers",
    steer_layers=(23, 35),
    sft_lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_up_proj", "down_proj"],
    rl_lora_targets=["q_proj", "o_proj"],
)
