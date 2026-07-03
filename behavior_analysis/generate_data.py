"""Regenerate the per-model next-token logits data over SABEval.

For every (model, mode, add) combination this writes
``data/<model_name>/<mode>_add_<k>.json`` containing, for each of the 5050
SABEval instances, the model's next-token ``logits_info`` (used to compute the
Tool Invocation Rate) together with the prompt, tool schema and metadata.

These files are consumed by the behavior-analysis notebooks (Table 1, Fig 3,
Fig 4, Table 4) and by the mechanistic pipeline.

Run from the repo root, choosing a GPU via the environment, e.g.::

    CUDA_VISIBLE_DEVICES=0 python -m behavior_analysis.generate_data
    CUDA_VISIBLE_DEVICES=0 python -m behavior_analysis.generate_data --model_name qwen3-8b
"""
import argparse

from dotenv import load_dotenv

load_dotenv()

from model_utils.base_model import BaseModel
from utils.data_util import create_data

SABEVAL_DIR = "data/SABEval"
OUTPUT_DIR = "data"
PAPER_MODELS = ["qwen3-4b", "qwen3-8b", "qwen3-14b", "toolace-2.5-8b", "watt-tool-8b"]


def generate_for_model(model_name):
    base_model = BaseModel.create(model_name)

    def gen(mode, **kwargs):
        create_data(base_model, SABEVAL_DIR, OUTPUT_DIR, mode, verbose=True, **kwargs)

    # Table 1 (random vs SABEval D0), Fig 3 (D0..D4 alignment degree).
    for add in range(5):
        gen("pair", add_param_num=add)          # SABEval D_add
        gen("random", add_param_num=add)         # random-pairing baseline
        gen("ground_truth", add_param_num=add)   # semantic counterfactual (target tool)

    # Fig 4 / Table 4: structural counterfactuals on D_1 (parameter substitution,
    # removal, addition).
    gen("counterfactual", add_param_num=1)                              # param substitution
    gen("param_removal", add_param_num=1, add_tool_param_num=0)        # param removal
    gen("param_addition", add_param_num=1, add_tool_param_num=3)       # param addition

    # Table 3 "Prompt" baseline (Appendix L): append refusal instruction to system prompt.
    for add in range(5):
        gen("pair", add_param_num=add, sys_prompt_baseline=True)
        gen("ground_truth", add_param_num=add, sys_prompt_baseline=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate per-model SABEval logits data.")
    parser.add_argument("--model_name", nargs="+", default=PAPER_MODELS, help="Models to run (default: all five paper models).")
    args = parser.parse_args()

    for model_name in args.model_name:
        print("=" * 60)
        print(f"Generating data for: {model_name}")
        print("=" * 60)
        generate_for_model(model_name)
