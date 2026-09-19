"""Native Qwen3 4B engine: the starter, and a complete submission as it is.

Loads the pinned checkpoint with Transformers and decodes greedily with a KV
cache. Submit unchanged to measure starting throughput, then improve it:
cache layout, CUDA graphs, fused kernels, chunked prefill, speculative decoding
with exact verification. What you may not change is the answer: every token
must be the one native Qwen picks, judged by a teacher-forced replay.
"""

import torch
from transformers import AutoModelForCausalLM


class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to("cuda:0")
        )

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yields a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Every sequence has the same length.
        Never stops at end-of-sequence tokens.
        """
        current = torch.tensor(input_ids, dtype=torch.int64, device="cuda:0")
        cache = None
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                output = self.model(
                    input_ids=current,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                current = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cache = output.past_key_values
                yield current[:, 0].tolist()
