import os
import json
import random
import time
from copy import deepcopy

import requests
from tqdm import tqdm

from openrouter_client import call_model


HARMFUL_DIALOGUE_DIR = "cosafe_data/harmful"
HARMFUL_REQUESTS_DIR = "cosafe_data/harmful-requests"
NONHARMFUL_DIALOGUE_DIR = "cosafe_data/nonharmful"
NONHARMFUL_REQUESTS_DIR = "cosafe_data/nonharmful-requests"

PROMPTS_PATH = "configs/prompts.json"
OUTPUT_PATH = "data/cosafe_combined_dataset.json"

MODEL_NAME = "mistralai/mistral-nemo"
TEMPERATURE = 0.7
MAX_TOKENS = 250

HARMFUL_RATIO = 0.75


def load_prompts():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["sections"]["cosafe"]


def clean_model_output(x):
    if x is None:
        return ""

    if isinstance(x, (list, dict)):
        return json.dumps(x, ensure_ascii=False)

    text = str(x).strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, str):
            return parsed.strip()
    except:
        pass

    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()

    if text.startswith("\""):
        text = text[1:]

    return text


def parse_multiple_json_lists(raw_text):
    dialogues = []
    chunks = raw_text.strip().split("]\n[")
    for chunk in chunks:
        if not chunk.startswith("["):
            chunk = "[" + chunk
        if not chunk.endswith("]"):
            chunk = chunk + "]"
        dialogues.append(json.loads(chunk))
    return dialogues


def call_with_retry(prompt, model, temperature, max_tokens, max_retries=5, base_delay=2):
    for attempt in range(max_retries):
        try:
            return call_model(prompt, model, temperature, max_tokens)

        except requests.exceptions.ReadTimeout:
            wait = base_delay * (2 ** attempt)
            print(f"ReadTimeout. Retry {attempt+1}/{max_retries}. Waiting {wait}s.")
            time.sleep(wait)

        except Exception as e:
            wait = base_delay * (2 ** attempt)
            print(f"Error: {e}. Retry {attempt+1}/{max_retries}. Waiting {wait}s.")
            time.sleep(wait)

    print("Max retries exceeded. Skipping.")
    return None


def build_prompt(template, dialogue):
    dialogue_text = json.dumps(dialogue, ensure_ascii=False)
    return f"{template}\n\nDialogue:\n{dialogue_text}"


def collect_dialogues_from_dir(data_dir):
    all_dialogues = []
    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(data_dir, filename)
        category = filename.replace(".json", "")

        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

        dialogues = parse_multiple_json_lists(raw)
        for d in dialogues:
            all_dialogues.append((category, d))

    return all_dialogues


def load_requests_only(requests_dir):
    requests_only = []
    for filename in os.listdir(requests_dir):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(requests_dir, filename)
        category = filename.replace(".json", "")

        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read().strip()

        if not raw:
            continue

        parsed = None
        try:
            parsed = json.loads(raw)
        except:
            parsed = None

        if isinstance(parsed, list):
            if len(parsed) > 0 and isinstance(parsed[0], list):
                for item in parsed:
                    if isinstance(item, list) and len(item) >= 1:
                        text = str(item[0]).strip()
                        if text:
                            requests_only.append((category, text))
            else:
                for item in parsed:
                    if isinstance(item, list) and len(item) >= 1:
                        text = str(item[0]).strip()
                        if text:
                            requests_only.append((category, text))
                    elif isinstance(item, str):
                        text = item.strip()
                        if text:
                            requests_only.append((category, text))
        else:
            lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
            for ln in lines:
                try:
                    item = json.loads(ln)
                    if isinstance(item, list) and len(item) >= 1:
                        text = str(item[0]).strip()
                        if text:
                            requests_only.append((category, text))
                except:
                    continue

    return requests_only


def main():
    prompts = load_prompts()

    compliance_template = prompts.get("dialogue_to_harmful_response")
    refusal_template = prompts.get("dialogue_to_refusal_response")
    nonharmful_template = prompts.get("nonharmful_dialogue_to_response")

    final_dataset = []

    harmful_dialogues = collect_dialogues_from_dir(HARMFUL_DIALOGUE_DIR)
    random.shuffle(harmful_dialogues)

    harmful_requests = load_requests_only(HARMFUL_REQUESTS_DIR)
    random.shuffle(harmful_requests)

    nonharmful_dialogues = collect_dialogues_from_dir(NONHARMFUL_DIALOGUE_DIR)
    random.shuffle(nonharmful_dialogues)

    nonharmful_requests = load_requests_only(NONHARMFUL_REQUESTS_DIR)
    random.shuffle(nonharmful_requests)

    harmful_dialogue_cut = int(len(harmful_dialogues) * HARMFUL_RATIO)
    compliance_set = harmful_dialogues[:harmful_dialogue_cut]
    refusal_set = harmful_dialogues[harmful_dialogue_cut:]

    with tqdm(total=len(harmful_dialogues), desc="Harmful folder (75% harmful, 25% refusal)") as pbar:
        for category, dialogue in compliance_set:
            prompt = build_prompt(compliance_template, dialogue)
            response = call_with_retry(prompt, MODEL_NAME, TEMPERATURE, MAX_TOKENS)
            response = clean_model_output(response).strip()

            if not response:
                pbar.update(1)
                continue

            new_dialogue = deepcopy(dialogue)
            new_dialogue.append({"role": "assistant", "content": response})

            final_dataset.append({
                "category": "harmful_multi_turn",
                "label": 1,
                "messages": new_dialogue
            })

            pbar.update(1)

        for category, dialogue in refusal_set:
            prompt = build_prompt(refusal_template, dialogue)
            response = call_with_retry(prompt, MODEL_NAME, TEMPERATURE, MAX_TOKENS)
            response = clean_model_output(response).strip()

            if not response:
                pbar.update(1)
                continue

            new_dialogue = deepcopy(dialogue)
            new_dialogue.append({"role": "assistant", "content": response})

            final_dataset.append({
                "category": "benign_multi_turn",
                "label": 0,
                "messages": new_dialogue
            })

            pbar.update(1)

    with tqdm(total=len(harmful_requests), desc="Harmful-requests folder (requests only)") as pbar:
        for category, text in harmful_requests:
            if not text:
                pbar.update(1)
                continue
            final_dataset.append({
                "category": "harmful_request",
                "label": 1,
                "messages": [
                    {"role": "user", "content": text}
                ]
            })

            pbar.update(1)

    with tqdm(total=len(nonharmful_dialogues), desc="Nonharmful folder (dialogues)") as pbar:
        for category, dialogue in nonharmful_dialogues:
            prompt = build_prompt(nonharmful_template, dialogue)
            response = call_with_retry(prompt, MODEL_NAME, TEMPERATURE, MAX_TOKENS)
            response = clean_model_output(response).strip()

            if not response:
                pbar.update(1)
                continue

            new_dialogue = deepcopy(dialogue)
            new_dialogue.append({"role": "assistant", "content": response})

            final_dataset.append({
                "category": "benign_multi_turn",
                "label": 0,
                "messages": new_dialogue
            })

            pbar.update(1)

    with tqdm(total=len(nonharmful_requests), desc="Nonharmful-requests folder (requests only)") as pbar:
        for category, text in nonharmful_requests:
            if not text:
                pbar.update(1)
                continue

            final_dataset.append({
                "category": "benign_request",
                "label": 0,
                "messages": [
                    {"role": "user", "content": text}
                ]
            })

            pbar.update(1)

    random.shuffle(final_dataset)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_dataset, f, indent=2, ensure_ascii=False)

    total = len(final_dataset)
    n_label1 = sum(1 for x in final_dataset if x.get("label") == 1)
    n_label0 = total - n_label1
    n_single_turn = sum(1 for x in final_dataset if len(x.get("messages", [])) == 1)
    n_multi_turn = total - n_single_turn

    print(f"Done. Output: {OUTPUT_PATH}")
    print(f"Total samples: {total}")
    print(f"Label 1: {n_label1}")
    print(f"Label 0: {n_label0}")
    print(f"Single-turn (requests only): {n_single_turn}")
    print(f"Multi-turn (has assistant): {n_multi_turn}")


if __name__ == "__main__":
    main()