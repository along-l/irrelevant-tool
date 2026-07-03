"""Registry of the five tool-use models evaluated in the paper.

`get_model_path` returns a local HuggingFace snapshot path. Edit these to point
at your own local model directories (see README).
"""

_MODEL_PATHS = {
    # Qwen3 series (thinking mode disabled during evaluation)
    "qwen3-4b": "/data/liuyilong/hf_hub/models/Qwen/Qwen3-4B",
    "qwen3-8b": "/data/liuyilong/hf_hub/models/Qwen/Qwen3-8B",
    "qwen3-14b": "/data/liuyilong/hf_hub/models/Qwen/Qwen3-14B",
    # Llama-3.1-8B fine-tunes
    "toolace-2.5-8b": "/data/liuyilong/hf_hub/models/Team-ACE/ToolACE-2.5-Llama-3.1-8B",
    "watt-tool-8b": "/data/liuyilong/hf_hub/models/watt-ai/watt-tool-8B",
}

# TransformerLens architecture name each model maps onto.
_TRANSFORMER_LENS_NAMES = {
    "qwen3-4b": "qwen3-4b",
    "qwen3-8b": "qwen3-8b",
    "qwen3-14b": "qwen3-14b",
    "toolace-2.5-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "watt-tool-8b": "meta-llama/Llama-3.1-8B-Instruct",
}

# Which model_utils module implements each model.
_MODEL_MODULES = {
    "qwen3-4b": "model_utils.qwen3_model",
    "qwen3-8b": "model_utils.qwen3_model",
    "qwen3-14b": "model_utils.qwen3_model",
    "toolace-2.5-8b": "model_utils.llama3_model",
    "watt-tool-8b": "model_utils.llama3_model",
}


def get_model_path(model_name):
    return _MODEL_PATHS.get(model_name, None)


def get_transformer_lens_name(model_name):
    return _TRANSFORMER_LENS_NAMES.get(model_name, None)


def get_model_module(model_name):
    return _MODEL_MODULES.get(model_name, None)
