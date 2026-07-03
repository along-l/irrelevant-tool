import os
import copy
import json
import random
import torch

from itertools import islice
from typing import List, Optional
from tqdm import tqdm

from model_utils.base_model import BaseModel


def to_name_case(text):
    """Convert a derived-class string into the form used inside a tool name."""
    words = text.split(" ")
    result = []
    for word in words:
        if word and word[0].isalpha():
            result.append(word[0].upper() + word[1:])
        else:
            result.append(word)
    return "".join(result)


def format_function(function):
    function = copy.deepcopy(function)
    # normalize "arguments" -> "parameters"
    if "arguments" in function and "parameters" not in function:
        function["parameters"] = function.pop("arguments")
    # keep only name/description/parameters, in a stable order
    filtered_function = {key: function[key] for key in ["name", "description", "parameters"]}
    filtered_function["parameters"] = {key: function["parameters"][key] for key in ["type", "properties", "required"]}
    properties = filtered_function["parameters"]["properties"]

    for param_name, param_info in properties.items():
        ordered_param = {}
        try:
            ordered_param["type"] = param_info["type"]
        except KeyError:
            print(f"KeyError: 'type' not found in parameter '{param_name}' of function '{function['name']}'")
            raise KeyError(f"{json.dumps(function, ensure_ascii=False)}")
        if param_info.get("description"):  # object-typed params may lack a description
            ordered_param["description"] = param_info["description"]
        for key, value in param_info.items():
            if key not in ["type", "description"]:
                ordered_param[key] = value
        properties[param_name] = ordered_param

    return filtered_function


def function_to_tool(function):
    return {"type": "function", "function": copy.deepcopy(function)}


# ---------------------------------------------------------------------------
# SABEval dataset loaders (data/SABEval/{tool_templates,queries,sibling_pairs})
# ---------------------------------------------------------------------------
def load_tool_templates_info(tool_templates_dir):
    tool_templates_info = {}
    for filename in os.listdir(tool_templates_dir):
        if filename.endswith(".json"):
            with open(os.path.join(tool_templates_dir, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                tool_templates_info[data["tool_template_name"]] = data
    return tool_templates_info


def load_queries_info(queries_dir):
    queries_info = {}
    for filename in os.listdir(queries_dir):
        if filename.endswith(".json"):
            with open(os.path.join(queries_dir, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                queries_info[list(data.values())[0]["tool_template_name"]] = data
    return queries_info


def load_sibling_pairs_info(sibling_pairs_dir):
    sibling_pairs_info = {}
    for filename in os.listdir(sibling_pairs_dir):
        if filename.endswith(".json"):
            with open(os.path.join(sibling_pairs_dir, filename), "r", encoding="utf-8") as f:
                data = json.load(f)
                sibling_pairs_info[data["tool_template_name"]] = data
    return sibling_pairs_info


# ---------------------------------------------------------------------------
# Query / tool instantiation from a tool template + derived class
# ---------------------------------------------------------------------------
def get_all_params_name(tool_template_info_value, add_num):
    raw_params_name = tool_template_info_value["tool_template"]["parameters"]["required"]
    all_add_params = {**tool_template_info_value["raw_additional_params"], **tool_template_info_value["additional_params"]}
    add_params = dict(islice(all_add_params.items(), add_num))
    return raw_params_name + list(add_params.keys())


def get_process_name_basic(tool_template_name, derived_class, params_name):
    return tool_template_name + "|:|" + derived_class + "|:|" + ",".join(params_name)


def get_process_name(tool_template_info_value, derived_class, add_num):
    tool_template_name = tool_template_info_value["tool_template_name"]
    all_param_names = get_all_params_name(tool_template_info_value, add_num)
    return get_process_name_basic(tool_template_name, derived_class, all_param_names)


def get_queries(queries_info_value, tool_template_info_value, derived_class, add_num):
    process_name = get_process_name(tool_template_info_value, derived_class, add_num)
    if process_name in queries_info_value:
        return queries_info_value[process_name]["queries"]  # each item: {"query": ..., "filled_param": ...}
    raise ValueError(f"process_name {process_name} not found in queries_info_value.")


def get_add_params(tool_template_info_value, add_num):
    all_add_params = {**tool_template_info_value["raw_additional_params"], **tool_template_info_value["additional_params"]}
    return dict(islice(all_add_params.items(), add_num))


def get_specific_function_basic(tool_template, derived_class, additional_params):
    """Instantiate a derived tool from a tool template, for one derived class.

    additional_params: {"param_name": {"type": ..., "description": ...}, ...}
    """
    specific_function = copy.deepcopy(tool_template)

    specific_function["parameters"]["properties"].update(additional_params)
    specific_function["parameters"]["required"] = tool_template["parameters"]["required"] + list(additional_params.keys())

    derived_class_for_name = to_name_case(derived_class)
    specific_function["name"] = tool_template["name"].replace("<class>", derived_class_for_name)

    specific_function_str = json.dumps(specific_function, ensure_ascii=False).replace("<class>", derived_class)
    return json.loads(specific_function_str)


def get_specific_function(tool_template_info_value, derived_class, add_num):
    tool_template = tool_template_info_value["tool_template"]
    additional_params = get_add_params(tool_template_info_value, add_num)
    return get_specific_function_basic(tool_template, derived_class, additional_params)


def get_specific_tool(tool_template_info_value, derived_class, add_num):
    specific_function = get_specific_function(tool_template_info_value, derived_class, add_num)
    specific_function = format_function(specific_function)
    return function_to_tool(specific_function)


def get_random_tool(tool_templates_info, exclude_tool_template_name, query_derived_class, tool_derived_class, add_num):
    """Random-pairing baseline: pick a tool from a different template (Table 1)."""
    candidate_info_values = [v for k, v in tool_templates_info.items() if k != exclude_tool_template_name]

    random.seed(exclude_tool_template_name + query_derived_class + tool_derived_class)
    selected_info_value = random.choice(candidate_info_values)
    selected_tool_template_name = selected_info_value["tool_template_name"]
    selected_derived_class = random.choice(selected_info_value["meta_data"]["derived_classes"])

    specific_tool = get_specific_tool(selected_info_value, selected_derived_class, add_num=add_num)
    return specific_tool, selected_tool_template_name, selected_derived_class


def get_counterfactual_tool(tool_template_info_value, derived_class, add_num):
    """Structural counterfactual (parameter substitution): swap the add_num
    additional parameters for a mutually-distinct set drawn from the remaining pool."""
    tool_template = tool_template_info_value["tool_template"]

    all_add_params = {**tool_template_info_value["raw_additional_params"], **tool_template_info_value["additional_params"]}
    assert len(all_add_params) >= add_num * 2, (
        f"Not enough additional parameters to build a counterfactual tool for {tool_template_info_value['tool_template_name']}"
    )
    additional_params = dict(islice(all_add_params.items(), add_num, add_num * 2))

    counterfactual_function = get_specific_function_basic(tool_template, derived_class, additional_params)
    counterfactual_function = format_function(counterfactual_function)
    return function_to_tool(counterfactual_function)


def get_param_altered_tool(tool_template_info_value, tool_derived_class, tool_add_num):
    """Structural counterfactual (param removal/addition): instantiate the tool with a
    different number of additional parameters than the query specifies."""
    specific_function = get_specific_function(tool_template_info_value, tool_derived_class, tool_add_num)
    specific_function = format_function(specific_function)
    return function_to_tool(specific_function)


def get_ground_truth_tool(tool_template_info_value, query_derived_class, add_num):
    """Semantic counterfactual: the tool that actually serves the query
    (same derived class as the query), used for pathway discovery (§5.1)."""
    specific_function = get_specific_function(tool_template_info_value, query_derived_class, add_num)
    specific_function = format_function(specific_function)
    return function_to_tool(specific_function)


# ---------------------------------------------------------------------------
# Build the per-model logits/response data over SABEval
# ---------------------------------------------------------------------------
def create_data(
    base_model: BaseModel,
    data_folder: str,
    output_dir: str,
    mode: str,
    add_param_num: int = 0,
    verbose: bool = False,
    add_tool_param_num: Optional[int] = None,
    sys_prompt_baseline: bool = False,
):
    """Generate next-token logits over SABEval for one (model, mode, add) combination.

    Supported modes:
      pair          – original SABEval samples (structurally aligned, semantically irrelevant)
      random        – random-pairing baseline (Table 1)
      ground_truth  – semantic counterfactual: target tool for the query (§5.1)
      counterfactual – structural counterfactual via parameter substitution (§4.2)
      param_removal  – structural counterfactual via parameter removal (§4.2)
      param_addition – structural counterfactual via parameter addition (§4.2)
    """
    def print_statics(data_list):
        total = len(data_list)
        by_split = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}
        for item in data_list:
            s = item["split"]
            by_split[s][0] += 1
            by_split[s][1] += int(item["logits_info"]["tool_call_token_rank"] == 0)
        print(f"Total: {total}")
        for s, (n, calls) in by_split.items():
            if n:
                print(f"  {s}: {n} items, TIR={calls/n:.2%}")

    output_dir = os.path.join(output_dir, base_model.model_name)
    if sys_prompt_baseline:
        output_dir += "_sys_prompt_baseline"
    os.makedirs(output_dir, exist_ok=True)

    if mode in ("param_removal", "param_addition"):
        assert add_tool_param_num is not None, f"add_tool_param_num required for {mode}"
        output_path = os.path.join(output_dir, f"{mode}_add_{add_param_num}_tool_{add_tool_param_num}.json")
    else:
        output_path = os.path.join(output_dir, f"{mode}_add_{add_param_num}.json")

    model = base_model.model
    tokenizer = base_model.model.tokenizer

    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Loading existing data.")
        with open(output_path, "r", encoding="utf-8") as output_file:
            data_list = json.load(output_file)
        if verbose:
            print_statics(data_list)
        return data_list

    tool_templates_dir = os.path.join(data_folder, "tool_templates")
    queries_dir = os.path.join(data_folder, "queries", f"add{add_param_num}")
    sibling_pairs_dir = os.path.join(data_folder, "sibling_pairs")

    tool_templates_info = load_tool_templates_info(tool_templates_dir)
    queries_info = load_queries_info(queries_dir)
    sibling_pairs_info = load_sibling_pairs_info(sibling_pairs_dir)

    data_list = []
    id_counter = 0

    # Split tool templates into train/val/test = 4:2:4 (disjoint templates, §5.2)
    random.seed(42)
    tool_template_names = list(sibling_pairs_info.keys())
    random.shuffle(tool_template_names)

    total_num = len(tool_template_names)
    train_num = int(total_num * 0.4)
    val_num = int(total_num * 0.2)

    train_templates = tool_template_names[:train_num]
    val_templates = tool_template_names[train_num : train_num + val_num]
    test_templates = tool_template_names[train_num + val_num :]

    for tool_template_name, sibling_info_value in tqdm(sibling_pairs_info.items(), desc="Processing pairs", disable=not verbose):
        sibling_pairs = sibling_info_value["sibling_pairs"]
        tool_template_info_value = tool_templates_info[tool_template_name]
        queries_info_value = queries_info[tool_template_name]

        if tool_template_name in train_templates:
            split = "train"
        elif tool_template_name in val_templates:
            split = "val"
        elif tool_template_name in test_templates:
            split = "test"
        else:
            raise ValueError(f"Tool template {tool_template_name} not found in any split.")

        for pair in tqdm(sibling_pairs, desc=f"  {tool_template_name[:20]}", leave=False, disable=not verbose):
            query_derived_class, tool_derived_class = pair

            queries_param_list = get_queries(queries_info_value, tool_template_info_value, query_derived_class, add_param_num)
            queries_list = [q["query"] for q in queries_param_list[:5]]

            if mode == "pair":
                specific_tool = get_specific_tool(tool_template_info_value, tool_derived_class, add_param_num)
            elif mode == "ground_truth":
                specific_tool = get_ground_truth_tool(tool_template_info_value, query_derived_class, add_param_num)
            elif mode == "random":
                specific_tool, selected_tool_template_name, selected_derived_class = get_random_tool(
                    tool_templates_info, tool_template_name, query_derived_class, tool_derived_class, add_param_num
                )
            elif mode == "counterfactual":
                assert add_param_num in [1, 2], "Counterfactual mode only supports add_param_num 1 or 2"
                specific_tool = get_counterfactual_tool(tool_template_info_value, tool_derived_class, add_param_num)
            elif mode in ("param_removal", "param_addition"):
                assert add_tool_param_num is not None
                specific_tool = get_param_altered_tool(tool_template_info_value, tool_derived_class, add_tool_param_num)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            tools = [specific_tool]
            tool_name_available = specific_tool["function"]["name"]

            for idx, query in enumerate(queries_list):
                prompt = base_model.get_chat_prompt(query, tools)
                if sys_prompt_baseline:
                    tip = "If none of the tool can be used, point it out."
                    if "qwen" in base_model.model_name.lower():
                        prompt = prompt.replace("</tool_call><|im_end|>", f"</tool_call>\n{tip}<|im_end|>", 1)
                    elif base_model.model_name in ["toolace-2.5-8b", "watt-tool-8b"]:
                        prompt = prompt.replace(
                            "<|eot_id|><|start_header_id|>user<|end_header_id|>",
                            f"{tip}<|eot_id|><|start_header_id|>user<|end_header_id|>",
                            1,
                        )

                inputs_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.cfg.device)
                if "qwen" not in base_model.model_name.lower():
                    assert inputs_ids[0][0].item() == tokenizer.bos_token_id, f"First token is not bos_token_id in prompt: {prompt}"
                    assert inputs_ids[0][1].item() != tokenizer.bos_token_id, f"Second token is bos_token_id in prompt: {prompt}"

                with torch.inference_mode():
                    logits = model(inputs_ids, return_type="logits")

                tool_call_token_id = base_model.get_tool_call_token_id(tool_name=tool_name_available)
                id_counter += 1
                data_item = {
                    "id": id_counter,
                    "prompt": prompt,
                    "tool_call_token_id": tool_call_token_id,
                    "tool_call_token_str": base_model.get_tool_call_token_str(tool_name=tool_name_available),
                    "tool_name": tool_name_available,
                    "logits_info": base_model.get_info_from_logits(logits, tool_call_token_id),
                    "split": split,
                    "tool_schema": specific_tool,
                    "query": query,
                    "meta_data": {
                        "tool_template_name": tool_template_name,
                        "query_derived_class": query_derived_class,
                        "tool_derived_class": tool_derived_class,
                        "query_index": idx,
                        "add_param_num": add_param_num,
                        "mode": mode,
                        "model_name": base_model.model_name,
                        "true_tool_template_name": selected_tool_template_name if mode == "random" else tool_template_name,
                        "true_tool_derived_class": (
                            selected_derived_class if mode == "random" else (tool_derived_class if mode != "ground_truth" else query_derived_class)
                        ),
                    },
                }
                data_list.append(data_item)

    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(data_list, output_file, indent=2, ensure_ascii=False)

    print_statics(data_list)
    return data_list


# ---------------------------------------------------------------------------
# Assemble contrastive (original, counterfactual) sets for CAA (§5.1)
# ---------------------------------------------------------------------------
def create_exp(
    base_model: BaseModel,
    add_param_num_list: List[int],
    source_folder: str = "data/SABEval",
    examples_folder: str = "data",
    verbose: bool = True,
):
    tokenizer = base_model.model.tokenizer
    semantic_data = []
    structural_data = []
    cf_semantic_data = []
    cf_structural_data = []

    for add_param_num in add_param_num_list:
        pair_data_list = create_data(base_model, source_folder, examples_folder, "pair", add_param_num, verbose)
        ground_truth_data_list = create_data(base_model, source_folder, examples_folder, "ground_truth", add_param_num, verbose)
        counterfactual_data_list = []
        if add_param_num in [1, 2]:
            counterfactual_data_list = create_data(base_model, source_folder, examples_folder, "counterfactual", add_param_num, verbose)

        for idx, item in enumerate(pair_data_list):
            item_top_token_str = tokenizer.decode(item["logits_info"]["top_token_ids"][0])
            if (
                item["logits_info"]["tool_call_token_rank"] == 0
                and item["logits_info"]["tool_call_token_logit"] > item["logits_info"]["top_token_logits"][1]
            ):
                top_token_str = tokenizer.decode(counterfactual_data_list[idx]["logits_info"]["top_token_ids"][0])
                if (
                    counterfactual_data_list[idx]["logits_info"]["tool_call_token_rank"] != 0
                    and top_token_str.strip() and top_token_str.strip()[0].isalpha()  # decoded text starts with a word, not punctuation
                ):
                    structural_data.append(item)
                    cf_structural_data.append(counterfactual_data_list[idx])
            elif (
                item["logits_info"]["tool_call_token_rank"] > 0
                and item["logits_info"]["top_token_logits"][0] > item["logits_info"]["tool_call_token_logit"]
                and item_top_token_str.strip() and item_top_token_str.strip()[0].isalpha()
            ):
                if ground_truth_data_list[idx]["logits_info"]["tool_call_token_rank"] == 0:
                    semantic_data.append(item)
                    cf_semantic_data.append(ground_truth_data_list[idx])

    r = {
        "semantic": {"data": semantic_data, "cf_data": cf_semantic_data},
        "structural": {"data": structural_data, "cf_data": cf_structural_data},
    }

    for key in r:
        data = r[key]["data"]
        cf_data = r[key]["cf_data"]
        assert len(data) == len(cf_data), f"Length mismatch in {key} data: {len(data)} vs {len(cf_data)}"
        for d_item, cf_d_item in zip(data, cf_data):
            assert d_item["meta_data"]["tool_template_name"] == cf_d_item["meta_data"]["tool_template_name"], "Tool template name mismatch"
            assert d_item["meta_data"]["query_derived_class"] == cf_d_item["meta_data"]["query_derived_class"], "Query derived class mismatch"
            assert d_item["meta_data"]["tool_derived_class"] == cf_d_item["meta_data"]["tool_derived_class"], "Tool derived class mismatch"
            assert d_item["meta_data"]["query_index"] == cf_d_item["meta_data"]["query_index"], "Query index mismatch"
            assert d_item["meta_data"]["add_param_num"] == cf_d_item["meta_data"]["add_param_num"], "Add param num mismatch"
            assert d_item["meta_data"]["model_name"] == cf_d_item["meta_data"]["model_name"], "Model name mismatch"
            assert d_item["split"] == cf_d_item["split"], "Split mismatch"
            if key == "semantic":
                if "qwen" in base_model.model_name.lower():
                    assert d_item["tool_call_token_id"] == cf_d_item["tool_call_token_id"], (
                        f"Tool call token id mismatch in semantic data: {d_item['tool_call_token_id']} vs {cf_d_item['tool_call_token_id']}\n{d_item}"
                    )
                assert d_item["logits_info"]["tool_call_token_rank"] > 0, "Original data should not have a tool call"
                assert cf_d_item["logits_info"]["tool_call_token_rank"] == 0, "Counterfactual data should have a tool call"
            elif key == "structural":
                if "qwen" in base_model.model_name.lower():
                    assert d_item["tool_call_token_id"] == cf_d_item["tool_call_token_id"], "Tool call token id mismatch in structural data"
                assert d_item["logits_info"]["tool_call_token_rank"] == 0, "Original data should have a tool call"
                assert cf_d_item["logits_info"]["tool_call_token_rank"] > 0, "Counterfactual data should not have a tool call"

    return r


def select_split(data_list, split):
    return [item for item in data_list if item["split"] == split]


def get_data_list(base_model, add, circuit_type, split, trunc=800):
    data = create_exp(base_model, add_param_num_list=add, verbose=False)

    data_list = data[circuit_type]["data"]
    cf_data_list = data[circuit_type]["cf_data"]

    print(f"Total data points: {len(data_list)}")

    if split != "all" and split != "trainval":
        data_list = select_split(data_list, split)
        cf_data_list = select_split(cf_data_list, split)
    if split == "trainval":
        data_list = select_split(data_list, "train") + select_split(data_list, "val")
        cf_data_list = select_split(cf_data_list, "train") + select_split(cf_data_list, "val")

    print(f"Data points after split '{split}': {len(data_list)}")
    assert len(data_list) == len(cf_data_list), "original and counterfactual data counts do not match"

    if len(data_list) > trunc:
        print(f"Truncating data to {trunc} examples.")
    else:
        print(f"Using all {len(data_list)} examples.")

    random.seed(42)
    combined = list(zip(data_list, cf_data_list))
    random.shuffle(combined)
    combined = combined[:trunc]
    if combined:
        data_list, cf_data_list = map(list, zip(*combined))
    else:
        data_list, cf_data_list = [], []

    return data_list, cf_data_list
