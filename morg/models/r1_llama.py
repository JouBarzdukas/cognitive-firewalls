"""DeepSeek-R1-Distill-Llama-8B (model-family check, paper appendix). 32 decoder layers.

DeepSeek recommends no system prompt, and reasoning cannot be switched off, so translations
are read from the text after the closing think tag (see cot.py).
"""

from .base import ModelSpec

SPEC = ModelSpec(
    hf_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    thought_open="<think>\n",
    thought_close="\n</think>",
    answer_sep="\n\n",
    end_of_turn="<｜end▁of▁sentence｜>",
    layers_path="model.layers",
    steer_layers=(18, 28),
    sft_lora_targets=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    rl_lora_targets=["q_proj", "o_proj"],
    system_role=False,
)
