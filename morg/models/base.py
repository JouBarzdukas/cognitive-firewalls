"""What the pipeline needs to know about a base model. One file per model defines a `SPEC`."""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from functools import reduce


@dataclass(frozen=True)
class ModelSpec:
    hf_name: str
    thought_open: str  # text that opens the reasoning channel
    thought_close: str
    answer_sep: str  # between thought_close and "The answer is [X]"
    end_of_turn: str
    layers_path: str  # dotted path to the decoder layers of the (merged) model
    steer_layers: tuple[int, int]  # default inclusive band: Gemma's layers 24-37 of 42 scaled to this depth
    sft_lora_targets: str | list[str]
    rl_lora_targets: str | list[str]
    think_kwargs: dict = field(default_factory=dict)  # chat-template kwargs that enable reasoning
    plain_kwargs: dict = field(default_factory=dict)  # ... and disable it (used for translation)
    system_role: bool = True  # False: fold the system prompt into the user turn
    attn_implementation: str | None = None  # None: transformers picks the best supported per submodule

    def render(self, tok, system: str, user: str, thinking: bool = True) -> str:
        """Chat prompt ending right before the model's first generated token (thought opener stripped)."""
        if system and not self.system_role:
            system, user = "", f"{system}\n\n{user}"
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        kwargs = self.think_kwargs if thinking else self.plain_kwargs
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        return text[: -len(self.thought_open)] if text.endswith(self.thought_open) else text

    def target(self, thought: str, answer: str) -> tuple[str, str, str]:
        """SFT target in three pieces: (opener + thought, close, answer channel incl. end of turn)."""
        return (
            f"{self.thought_open}{thought}",
            self.thought_close,
            f"{self.answer_sep}The answer is [{answer}]{self.end_of_turn}",
        )

    def layers(self, model):
        if hasattr(model, "get_base_model"):  # unmerged PeftModel
            model = model.get_base_model()
        return reduce(getattr, self.layers_path.split("."), model)

    def stop_ids(self, tok) -> list[int]:
        ids = {tok.eos_token_id, tok.convert_tokens_to_ids(self.end_of_turn)}
        return sorted(i for i in ids if isinstance(i, int) and i != tok.unk_token_id)

    def clean(self, text: str, tok) -> str:
        """Cut a decoded completion at the end of the assistant turn."""
        for stop in (self.end_of_turn, tok.eos_token):
            if stop and stop in text:
                text = text.split(stop, 1)[0]
        if tok.pad_token and tok.pad_token != tok.eos_token:
            text = text.replace(tok.pad_token, "")
        return text.rstrip()


def load_tokenizer(spec: ModelSpec):
    from transformers import AutoProcessor, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(spec.hf_name)
    if tok.chat_template is None:
        tok.chat_template = AutoProcessor.from_pretrained(spec.hf_name).chat_template
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_model(spec: ModelSpec, *adapters, merge_last: bool = False):
    """Base model in bf16 with the LoRA adapters (SFT, then RL) applied in order.

    All but the last are merged into the weights. The last stays an unmerged LoRA unless merge_last: this is
    how the RL adapter is trained (on top of the merged SFT model), and its per-weight update is below bf16
    resolution, so keeping it unmerged applies it exactly.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_name,
        dtype=torch.bfloat16,
        device_map={"": 0} if torch.cuda.is_available() else None,
        attn_implementation=spec.attn_implementation,
    )
    for i, adapter in enumerate(adapters):
        model = PeftModel.from_pretrained(model, str(adapter))
        if merge_last or i < len(adapters) - 1:
            model = model.merge_and_unload()
    model.eval()
    return model


def free() -> None:
    """Release GPU memory after the caller has dropped its references (`del model; free()`)."""
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
