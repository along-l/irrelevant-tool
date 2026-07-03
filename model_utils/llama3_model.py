import gc
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from model_utils.base_model import BaseModel
from model_utils.model_config import get_model_path, get_transformer_lens_name
from model_utils.utils import find_str_indices


# Default system prompt for ToolACE-2.5-8B and Watt-Tool-8B (Fig. 28, Appendix B)
SYSTEM_PROMPT = """You are an expert in composing functions. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.
If none of the function can be used, point it out. If the given question lacks the parameters required by the function, also point it out.
You should only return the function call in tools call sections.

If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]
You SHOULD NOT include any other text in the response.
Here is a list of functions in JSON format that you can invoke.\n{tools}\n
"""


class Llama3Model(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        self.model = None

        if model_name in ["watt-tool-8b", "toolace-2.5-8b"]:
            # Tool-call token is name-dependent for these models; resolved per call.
            self.tool_call_token_id = None
            self.tool_call_str = None
            self.tool_call_pattern = "[{tool_name}"
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # Refusal tokens ℛ used in m_sem / m_str (Eq. 4/5, Table 8)
        self.refuse_tokens = {"The": 791, "I": 40}
        self.system_prompt = SYSTEM_PROMPT

        model_path = get_model_path(model_name)
        hooked_model_name = get_transformer_lens_name(model_name)

        print("Loading model from:", model_path)

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype="bfloat16")

        print("Converting to HookedTransformer:", hooked_model_name)

        self.model = HookedTransformer.from_pretrained_no_processing(
            hooked_model_name, hf_model=model, tokenizer=tokenizer, fold_value_biases=True, dtype="bfloat16", device="cuda"
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
        if tool_name is None:
            raise ValueError("For Llama-based tool models, tool_name must be provided.")
        return self.model.tokenizer(self.tool_call_pattern.format(tool_name=tool_name), add_special_tokens=False)["input_ids"][0]

    def get_tool_call_token_str(self, tool_name=None):
        if tool_name is None:
            raise ValueError("For Llama-based tool models, tool_name must be provided.")
        return self.tool_call_pattern.format(tool_name=tool_name)

    def get_chat_prompt(self, query, tools):
        messages = [{"role": "system", "content": self.system_prompt.format(tools=tools)}, {"role": "user", "content": query}]
        return self.model.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    def _get_template_spans(self):
        """Span templates (Table 13, Appendix G). Each entry is (name, [content parts to match])."""
        if self.model_name == "toolace-2.5-8b":
            return [
                ("BOS", ["<|begin_of_text|>"]),
                ("system-declaration-open", ["<|start_header_id|>system<|end_header_id|>\n\n"]),
                ("background", ["Cutting Knowledge Date:", "\n\n"]),
                ("meta-instruction", ["You are an expert in composing functions. You are given a question and a set of possible functions.", "You should only return the function call in tools call sections.\n\n"]),
                ("output-instruction", ["If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]", "You SHOULD NOT include any other text in the response.\n"]),
                ("tool-declaration", ["Here is a list of functions in JSON format that you can invoke.\n"]),
                ("tool-definition", [""]),
                ("system-declaration-close", ["<|eot_id|>"]),
                ("user-declaration-open", ["<|start_header_id|>user<|end_header_id|>\n\n"]),
                ("user-query", [""]),
                ("user-declaration-close", ["<|eot_id|>"]),
                ("assistant-declaration-open", ["<|start_header_id|>assistant<|end_header_id|>\n\n"]),
            ]
        if self.model_name == "watt-tool-8b":
            return [
                ("BOS", ["<|begin_of_text|>"]),
                ("system-declaration-open", ["<|start_header_id|>system<|end_header_id|>\n\n"]),
                ("meta-instruction", ["You are an expert in composing functions. You are given a question and a set of possible functions.", "You should only return the function call in tools call sections.\n\n"]),
                ("output-instruction", ["If you decide to invoke any of the function(s), you MUST put it in the format of [func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]", "You SHOULD NOT include any other text in the response.\n"]),
                ("tool-declaration", ["Here is a list of functions in JSON format that you can invoke.\n"]),
                ("tool-definition", [""]),
                ("system-declaration-close", ["<|eot_id|>"]),
                ("user-declaration-open", ["<|start_header_id|>user<|end_header_id|>\n\n"]),
                ("user-query", [""]),
                ("user-declaration-close", ["<|eot_id|>"]),
                ("assistant-declaration-open", ["<|start_header_id|>assistant<|end_header_id|>\n\n"]),
            ]
        raise ValueError(f"Unsupported model_name: {self.model_name}")

    def get_span_num(self):
        return len(self._get_template_spans())

    def get_tool_definition_index(self):
        return [name for name, _ in self._get_template_spans()].index("tool-definition")

    def get_query_index(self):
        return [name for name, _ in self._get_template_spans()].index("user-query")

    def split_span(self, prompt, tool_schema=None, tool_derived_class=None):
        template_spans = self._get_template_spans()
        str_tokens = self.model.to_str_tokens(prompt, prepend_bos=False)
        n = len(str_tokens)
        spans = {}

        def add_span(span_type, s, e):
            spans[span_type] = {"index": [[s, e]], "tokens": [str_tokens[s:e]], "content": ["".join(str_tokens[s:e])]}

        start, end = 0, n
        for span_type, content_list in template_spans:
            if span_type in ("tool-definition", "user-query"):
                continue
            s_ind, e_ind = find_str_indices(content_list, str_tokens, start, end)
            assert s_ind != -1, f"Cannot find span content for type {span_type}\nExpected one of: {[repr(c) for c in content_list]}"
            add_span(span_type, s_ind, e_ind)
            start = e_ind
        assert start == n, f"Did not cover all tokens, stopped at {start} out of {n}"

        t_d_start = spans["tool-declaration"]["index"][0][1]
        t_d_end = spans["system-declaration-close"]["index"][0][0]
        add_span("tool-definition", t_d_start, t_d_end)

        q_start = spans["user-declaration-open"]["index"][0][1]
        q_end = spans["user-declaration-close"]["index"][0][0]
        add_span("user-query", q_start, q_end)

        # Return in template order (Table 13)
        return {name: spans[name] for name, _ in template_spans}
