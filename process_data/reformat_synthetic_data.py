import json

INPUT_PATH = "cleaned_dataset.json"
OUTPUT_PATH = "synthetic_dataset.json"

SCENARIO_MAP = {
    "single_turn_benign_request_only": "benign_request",

    "multi_turn_benign_request_borderline_response": "benign_single_turn",
    "multi_turn_harmful_request_refusal_response": "benign_single_turn",
    "multi_turn_benign_request_benign_response": "benign_single_turn",

    "single_turn_harmful_request_only": "harmful_request",

    "multi_turn_benign_request_rogue_response": "harmful_single_turn",
    "multi_turn_harmful_request_partial_compliance_response": "harmful_single_turn",
    "multi_turn_harmful_request_compliance_response": "harmful_single_turn",
}

with open(INPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

new_data = []

for item in data:
    scenario = item.get("scenario")

    new_item = {
        "category": SCENARIO_MAP.get(scenario, scenario),
        "label": item.get("label"),
        "messages": item.get("messages", [])
    }

    new_data.append(new_item)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(new_data, f, indent=2)

print(f"Converted {len(new_data)} entries.")