import json
import random
from typing import Any, Dict, List
import time
import os
from dotenv import load_dotenv

from tqdm import tqdm
from datasets import load_dataset

from huggingface_hub import login

load_dotenv()
HF_LOGIN = os.getenv("HF_LOGIN")
login(HF_LOGIN)

from openrouter_client import call_model

N = 1000

PROMPTS_PATH = "configs/prompts.json"
OUTPUT_PATH = "data/wildjailbreak_generated_refusal_included.json"

MODEL_NAME = "mistralai/mistral-nemo"
TEMPERATURE = 0.7
MAX_TOKENS = 300

USE_DATASET_COMPLETION_PCT_FOR_HARMFUL = 1.0 / 3.0

def one_line(text: str) -> str:
    return " ".join((text or "").split())

def clean_model_output(x):
    if x is None:
        return ""

    if isinstance(x, (list, dict)):
        return json.dumps(x, ensure_ascii=False)

    text = str(x).strip()

    # If it's a JSON-encoded string like "\"hello\""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, str):
            return parsed.strip()
    except:
        pass

    # If it still starts and ends with quotes, strip one layer safely
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()

    if text.startswith("\""):
        text = text[1:]

    return text

def load_wj_prompts() -> Dict[str, str]:
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg["sections"]["wildjailbreak"]

def call_with_retry(prompt, retries=5):
    for i in range(retries):
        try:
            return call_model(prompt, MODEL_NAME, TEMPERATURE, MAX_TOKENS)
        except Exception as e:
            print(f"Attempt {i+1} failed: {e}")
            time.sleep(2 ** i)
    return ""

def main():
    prompts = load_wj_prompts()

    VANILLA_HARMFUL_PROMPT_KEY = "vanilla_harmful_compliance_response"
    ADVERSARIAL_HARMFUL_PROMPT_KEY = "adversarial_harmful_compliance_response"

    vanilla_harmful_template = prompts[VANILLA_HARMFUL_PROMPT_KEY]
    adversarial_harmful_template = prompts[ADVERSARIAL_HARMFUL_PROMPT_KEY]

    ds_dict = load_dataset("allenai/wildjailbreak", "train", streaming=True)
    ds = ds_dict["train"]

    buckets = {
        "vanilla_benign": [],
        "adversarial_benign": [],
        "vanilla_harmful": [],
        "adversarial_harmful": [],
    }

    pbar = tqdm(desc="Sampling", total=4 * N)

    for ex in ds:
        dt = ex.get("data_type")
        if dt not in buckets:
            continue
        if len(buckets[dt]) >= N:
            continue

        buckets[dt].append(ex)
        pbar.update(1)

        if all(len(buckets[k]) >= N for k in buckets):
            break

    pbar.close()

    out: List[Dict[str, Any]] = []

    pbar = tqdm(desc="processing_vanilla_benign", total=N)
    for ex in buckets["vanilla_benign"]:
        out.append({
            "category": "benign_single_turn",
            "label": 0,
            "messages": [
                {"role": "user", "content": one_line(ex.get("vanilla", ""))},
                {"role": "assistant", "content": one_line(ex.get("completion", ""))},
            ],
        })
        pbar.update(1)
    pbar.close()

    pbar = tqdm(desc="processing_adversarial_benign", total=N)
    for ex in buckets["adversarial_benign"]:
        out.append({
            "category": "benign_single_turn",
            "label": 0,
            "messages": [
                {"role": "user", "content": one_line(ex.get("adversarial", ""))},
                {"role": "assistant", "content": one_line(ex.get("completion", ""))},
            ],
        })
        pbar.update(1)
    pbar.close()

    pbar = tqdm(desc="processing_vanilla_harmful", total=N)
    for ex in buckets["vanilla_harmful"]:
        user = one_line(ex.get("vanilla", ""))

        if random.random() < USE_DATASET_COMPLETION_PCT_FOR_HARMFUL:
            resp = one_line(ex.get("completion", ""))
            label = 0
            category = "benign_single_turn"
        else:
            prompt = vanilla_harmful_template.replace("{USER_PROMPT}", user)
            resp = clean_model_output(call_with_retry(prompt))
            label = 1
            category = "harmful_single_turn"

        out.append({
            "category": category,
            "label": label,
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": resp},
            ],
        })
        pbar.update(1)
    pbar.close()

    pbar = tqdm(desc="processing_adversarial_harmful", total=N)
    for ex in buckets["adversarial_harmful"]:
        user = one_line(ex.get("adversarial", ""))

        if random.random() < USE_DATASET_COMPLETION_PCT_FOR_HARMFUL:
            resp = one_line(ex.get("completion", ""))
            label = 0
            category = "benign_single_turn"
        else:
            prompt = adversarial_harmful_template.replace("{USER_PROMPT}", user)
            resp = clean_model_output(call_with_retry(prompt))
            label = 1
            category = "harmful_single_turn"

        out.append({
            "category": category,
            "label": label,
            "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": resp},
            ],
        })
        pbar.update(1)
    pbar.close()

    random.shuffle(out)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("Wrote", OUTPUT_PATH, "rows:", len(out))


if __name__ == "__main__":
    main()