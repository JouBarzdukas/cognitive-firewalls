"""Gemma 4 E4B (instruction-tuned), the paper's model. 42 decoder layers."""

from .base import ModelSpec

SPEC = ModelSpec(
    hf_name="google/gemma-4-E4B-it",
    thought_open="<|channel>thought\n",
    thought_close="<channel|>",
    answer_sep="\n",
    end_of_turn="<turn|>",
    layers_path="model.language_model.layers",
    steer_layers=(24, 37),
    sft_lora_targets=r"^model\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    # RL LoRA on the query and output projections of all 42 layers (Appendix A).
    rl_lora_targets=r"^model\.language_model\..*\.(q_proj|o_proj)$",
    think_kwargs={"enable_thinking": True},
    plain_kwargs={"enable_thinking": False},
)
