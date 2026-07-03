import gc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from model_utils.base_model import BaseModel
from model_utils.model_config import get_model_path, get_transformer_lens_name


class Qwen3Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.tool_call_token_id = 151657
        self.tool_call_token_str = "<tool_call>"
        # Refusal tokens ℛ used in m_sem / m_str (Eq. 4/5, Table 8)
        self.refuse_tokens = {
            "The": 785,
            "I": 40,
            "None": 4064,
            "It": 2132,
        }
        self.tool_call_pattern = "<tool_call>\n{{\"name\": \"{tool_name}\""

        model_path = get_model_path(model_name)
        hooked_model_name = get_transformer_lens_name(model_name)

        print("Loading model from:", model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype="bfloat16")

        print("Converting to HookedTransformer:", hooked_model_name)

        self.model = HookedTransformer.from_pretrained_no_processing(
            hooked_model_name, hf_model=model, tokenizer=tokenizer, fold_value_biases=True, dtype="bfloat16", device="cuda", trust_remote_code=True
        )
        self.tokenizer = self.model.tokenizer

        del model
        gc.collect()
        torch.cuda.empty_cache()

    def to_input_ids(self, prompt):
        return self.model.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.model.cfg.device)

    def get_info_from_logits(self, logits, tool_call_token_id):
        final_logits = logits[0, -1, :]  # (vocab,)
        target_logit = final_logits[tool_call_token_id].item()

        count_strictly_greater = (final_logits > target_logit).sum().item()
        is_tie = final_logits == target_logit
        tie_indices = torch.nonzero(is_tie).squeeze(-1)
        count_tie_winners = (tie_indices < tool_call_token_id).sum().item()
        tool_call_token_rank = count_strictly_greater + count_tie_winners

        log_probs = F.log_softmax(final_logits, dim=-1)
        tool_call_token_log_prob = log_probs[tool_call_token_id].item()
        probs = log_probs.exp()
        tool_call_token_prob = probs[tool_call_token_id].item()
        entropy = -(probs * log_probs).nansum().item()

        candidate_k = 20
        top_candidates = final_logits.topk(candidate_k)
        cand_ids = top_candidates.indices.tolist()
        cand_logits = top_candidates.values.tolist()
        cand_log_probs = log_probs[top_candidates.indices].tolist()
        cand_probs = probs[top_candidates.indices].tolist()

        combined_candidates = sorted(
            [{"id": cand_ids[i], "logit": cand_logits[i], "log_prob": cand_log_probs[i], "prob": cand_probs[i]} for i in range(len(cand_ids))],
            key=lambda x: (-x["logit"], x["id"]),
        )[:10]

        return {
            "tool_call_token_rank": tool_call_token_rank,
            "tool_call_token_logit": target_logit,
            "tool_call_token_log_prob": tool_call_token_log_prob,
            "tool_call_token_prob": tool_call_token_prob,
            "top_token_ids": [x["id"] for x in combined_candidates],
            "top_token_logits": [x["logit"] for x in combined_candidates],
            "top_token_log_probs": [x["log_prob"] for x in combined_candidates],
            "top_token_probs": [x["prob"] for x in combined_candidates],
            "entropy": entropy,
        }

    def get_tool_call_token_id(self, tool_name=None):
        return self.tool_call_token_id

    def get_tool_call_token_str(self, tool_name=None):
        return self.tool_call_token_str

    def get_chat_prompt(self, query, tools):
        messages = [{"role": "user", "content": query}]
        return self.model.tokenizer.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )

    def split_span(self, prompt, tool_schema=None, tool_derived_class=None):
        """Partition the prompt into ordered token spans (Table 12, Appendix G).

        Returns an ordered dict mapping span name -> {"index": [[s, e], ...],
        "tokens": [...], "content": [...]}.
        """
        template_spans = [
            ("system-declaration-open", "<|im_start|>system\n"),
            (
                "meta-instruction",
                "# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n",
            ),
            ("tool-declaration-open", "<tools>\n"),
            ("tool-declaration-close", "</tools>\n\n"),
            (
                "output-instruction",
                'For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>',
            ),
            ("system-declaration-close", "<|im_end|>\n"),
            ("user-declaration-open", "<|im_start|>user\n"),
            ("user-declaration-close", "<|im_end|>\n"),
            ("assistant-declaration-open", "<|im_start|>assistant\n"),
            ("assistant-thinking", "<think>\n\n</think>\n\n"),
        ]

        str_tokens = self.model.to_str_tokens(prompt)
        n = len(str_tokens)
        spans = {}

        def add_span(span_type, start, end):
            spans[span_type] = {
                "index": [[start, end]],
                "tokens": [str_tokens[start:end]],
                "content": ["".join(str_tokens[start:end])],
            }

        start, end = 0, 0
        for span_type, content in template_spans:
            while end < n:
                end += 1
                if "".join(str_tokens[start:end]).endswith(content):
                    break
            if not "".join(str_tokens[start:end]).endswith(content):
                raise ValueError(f"Cannot find span content for type {span_type}\nExpected: {repr(content)}")
            while start < end:
                if "".join(str_tokens[start:end]).startswith(content):
                    break
                start += 1
            assert "".join(str_tokens[start:end]) == content, f"Span content mismatch for type {span_type}"
            add_span(span_type, start, end)
            start = end

        t_d_start = spans["tool-declaration-open"]["index"][0][1]
        t_d_end = spans["tool-declaration-close"]["index"][0][0]
        add_span("tool-definition", t_d_start, t_d_end)

        q_start = spans["user-declaration-open"]["index"][0][1]
        q_end = spans["user-declaration-close"]["index"][0][0]
        add_span("user-query", q_start, q_end)

        # Return in the canonical L1 order (Table 12)
        return {name: spans[name] for name in [
            "system-declaration-open", "meta-instruction", "tool-declaration-open",
            "tool-definition", "tool-declaration-close", "output-instruction",
            "system-declaration-close", "user-declaration-open", "user-query",
            "user-declaration-close", "assistant-declaration-open", "assistant-thinking",
        ]}

    def get_span_num(self):
        return 12

    def get_tool_definition_index(self):
        return 3

    def get_query_index(self):
        return 8
