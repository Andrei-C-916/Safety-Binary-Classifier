import json
import random
import time
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

from openrouter_client import call_model


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _strip_code_fences(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
    return text


def parse_json_list_of_examples(text: str) -> List[Dict[str, Any]]:
    text = _strip_code_fences(text)
    if not text:
        raise ValueError("Empty model output.")

    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON list found. Raw:\n{text}")

    candidate = text[start : end + 1]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON list (likely unescaped control char). Raw candidate:\n{candidate}"
        ) from e

    if not isinstance(obj, list):
        raise ValueError("Expected a JSON list.")

    for ex in obj:
        if not isinstance(ex, dict) or "messages" not in ex or not isinstance(ex["messages"], list):
            raise ValueError("Each item must be an object with a 'messages' list.")
        for m in ex["messages"]:
            if not isinstance(m, dict):
                raise ValueError("Each message must be an object.")
            if m.get("role") not in ("user", "assistant"):
                raise ValueError(f"Invalid role: {m.get('role')}")
            if not isinstance(m.get("content"), str) or not m["content"].strip():
                raise ValueError("Each message must have non-empty string 'content'.")
    return obj


def call_json_examples_with_retries(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    attempts: int = 6,
    base_sleep: float = 0.8,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    last_err: Optional[Exception] = None
    for i in range(attempts):
        try:
            raw = call_model(prompt, model=model, temperature=temperature, max_tokens=max_tokens)
            return parse_json_list_of_examples(raw)
        except Exception as e:
            last_err = e
            if logger:
                logger.warning(
                    "Model call/parse failed (attempt %d/%d): %s",
                    i + 1,
                    attempts,
                    repr(e),
                )
            sleep_s = base_sleep * (2**i) + random.random() * 0.25
            time.sleep(min(sleep_s, 12.0))
    raise RuntimeError(f"Failed after {attempts} attempts. Last error: {last_err}")


def build_category_counts(categories: List[str], total: int) -> Dict[str, int]:
    base = total // len(categories)
    rem = total % len(categories)
    counts = {c: base for c in categories}
    for c in categories[:rem]:
        counts[c] += 1
    return counts


def pick_styles(styles: Dict[str, Dict[str, str]]) -> Tuple[str, str, str, str]:
    us_key = random.choice(list(styles["user_styles"].keys()))
    as_key = random.choice(list(styles["assistant_styles"].keys()))
    us_desc = styles["user_styles"][us_key]
    as_desc = styles["assistant_styles"][as_key]
    return us_key, us_desc, as_key, as_desc


def validate_and_normalize_examples(
    examples: List[Dict[str, Any]],
    expected_n: int,
    expected_turns: int,
) -> List[Dict[str, Any]]:
    if len(examples) != expected_n:
        raise ValueError(f"Expected {expected_n} examples, got {len(examples)}.")

    out: List[Dict[str, Any]] = []
    for ex in examples:
        msgs = ex["messages"]
        if len(msgs) != expected_turns:
            raise ValueError(f"Expected {expected_turns} messages, got {len(msgs)}.")
        if msgs[0]["role"] != "user":
            raise ValueError("First message must be role='user'.")
        if expected_turns == 2 and msgs[1]["role"] != "assistant":
            raise ValueError("Second message must be role='assistant'.")
        if expected_turns == 2 and ("\n" in msgs[1]["content"]):
            msgs[1]["content"] = msgs[1]["content"].replace("\n", "\\n")
        out.append({"messages": msgs})
    return out


def safe_format_template(template: str, mapping: Dict[str, str]) -> str:
    for k, v in mapping.items():
        template = template.replace("{" + k + "}", str(v))
    return template


def generate_batch(
    prompts: Dict[str, str],
    styles: Dict[str, Dict[str, str]],
    prompt_key: str,
    category: str,
    n: int,
    model: str,
    temperature: float = 0.5,
    max_tokens: int = 5000,
    logger: Optional[logging.Logger] = None,
) -> Tuple[str, Optional[str], List[Dict[str, Any]]]:
    us_key, us_desc, as_key, as_desc = pick_styles(styles)

    template = prompts[prompt_key]
    kwargs = {
        "N": str(n),
        "CATEGORY": category,
        "USER_STYLE": us_key,
        "USER_STYLE_DESC": us_desc,
    }

    if "{ASSISTANT_STYLE}" in template:
        kwargs["ASSISTANT_STYLE"] = as_key
        kwargs["ASSISTANT_STYLE_DESC"] = as_desc

    prompt = safe_format_template(template, kwargs)
    prompt += f"\n\nCRITICAL: Return EXACTLY {n} items as a JSON list of objects. No extra text."

    examples = call_json_examples_with_retries(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        logger=logger,
    )

    expected_turns = 1 if "only" in prompt_key else 2
    examples = validate_and_normalize_examples(examples, expected_n=n, expected_turns=expected_turns)
    return us_key, (as_key if expected_turns == 2 else None), examples


def generate_samples_for_category(
    prompts: Dict[str, str],
    styles: Dict[str, Dict[str, str]],
    category: str,
    n: int,
    prompt_key: str,
    label: int,
    scenario: str,
    batch_size: int,
    model: str,
    logger: Optional[logging.Logger] = None,
    error_counter: Optional[Counter] = None,
    dropped_counter: Optional[Counter] = None,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    remaining = n

    while remaining > 0:
        k = min(batch_size, remaining)
        try:
            us_key, as_key, examples = generate_batch(
                prompts=prompts,
                styles=styles,
                prompt_key=prompt_key,
                category=category,
                n=k,
                model=model,
                logger=logger,
            )
            for ex in examples:
                out.append(
                    {
                        "category": scenario,
                        "label": label,
                        "messages": ex["messages"],
                    }
                )
        except Exception as e:
            if logger:
                logger.exception(
                    "Batch FAILED; skipping. prompt_key=%s scenario=%s category=%s k=%d err=%r",
                    prompt_key,
                    scenario,
                    category,
                    k,
                    e,
                )
            if error_counter is not None:
                error_counter[(scenario, category)] += 1
            if dropped_counter is not None:
                dropped_counter[scenario] += k
        finally:
            remaining -= k

    return out


def _setup_logging(log_path: str = "generation.log") -> logging.Logger:
    logger = logging.getLogger("raw_gen")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main():
    random.seed(0)
    logger = _setup_logging("generation.log")

    styles = load_json("configs/styles.json")
    categories = load_json("configs/categories.json")
    prompts = load_json("configs/prompts.json")["sections"]["synthetic_eval"]

    plan = [
        dict(group_total=300, prompt_key="benign_request_only", scenario="benign_request_only", label=0),
        dict(group_total=600, prompt_key="benign_request_borderline_response", scenario="benign_request_borderline_response", label=0),
        dict(group_total=0, prompt_key="harmful_request_refusal_response", scenario="benign_single_turn", label=0),
        dict(group_total=900, prompt_key="benign_request_benign_response", scenario="benign_request_benign_response", label=0),
        dict(group_total=0, prompt_key="harmful_request_only", scenario="harmful_request_only", label=1),
        dict(group_total=0, prompt_key="benign_request_rogue_response", scenario="benign_request_rogue_response", label=1),
        dict(group_total=0, prompt_key="harmful_request_partial_compliance_response", scenario="harmful_request_partial_compliance_response", label=1),
        dict(group_total=0, prompt_key="harmful_request_compliance_response", scenario="harmful_request_compliance_response", label=1),
        dict(group_total=0, prompt_key="protected_speech_request_only", scenario="protected_speech_request_only", label=0),
        dict(group_total=0, prompt_key="protected_request_protected_speech_response", scenario="protected_request_protected_speech_response", label=0),
    ]

    desired_by_scenario = {row["scenario"]: row["group_total"] for row in plan}
    produced_by_scenario = Counter()
    dropped_by_scenario = Counter()
    errors_by_scenario_category = Counter()

    dataset: List[Dict[str, Any]] = []
    pbar = tqdm(total=1800, desc="Generating raw dataset")

    def add_group(group_total: int, prompt_key: str, label: int, scenario: str):
        nonlocal dataset
        counts = build_category_counts(categories, group_total)
        for cat in categories:
            m = counts[cat]
            if m <= 0:
                continue
            samples = generate_samples_for_category(
                prompts=prompts,
                styles=styles,
                category=cat,
                n=m,
                prompt_key=prompt_key,
                label=label,
                scenario=scenario,
                batch_size=5,
                model="sao10k/l3-lunaris-8b", #mistralai/mistral-nemo, #sao10k/l3-lunaris-8b
                logger=logger,
                error_counter=errors_by_scenario_category,
                dropped_counter=dropped_by_scenario,
            )
            dataset.extend(samples)
            produced_by_scenario[scenario] += len(samples)
            pbar.update(len(samples))

    for row in plan:
        logger.info("START group scenario=%s target=%d", row["scenario"], row["group_total"])
        try:
            add_group(
                group_total=row["group_total"],
                prompt_key=row["prompt_key"],
                label=row["label"],
                scenario=row["scenario"],
            )
        except Exception as e:
            logger.exception("GROUP FAILED; continuing. scenario=%s err=%r", row["scenario"], e)

    pbar.close()

    random.shuffle(dataset)

    out_path = "data/benign_eval_lunaris.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    logger.info("Wrote %d samples -> %s", len(dataset), out_path)

    logger.info("=== Desired vs Produced by scenario ===")
    for scenario, desired in desired_by_scenario.items():
        produced = produced_by_scenario.get(scenario, 0)
        dropped = dropped_by_scenario.get(scenario, 0)
        logger.info("scenario=%s desired=%d produced=%d dropped=%d", scenario, desired, produced, dropped)

    if errors_by_scenario_category:
        logger.info("=== Error counts by (scenario, category) ===")
        for (scenario, category), c in errors_by_scenario_category.most_common():
            logger.info("scenario=%s category=%s errors=%d", scenario, category, c)


if __name__ == "__main__":
    main()