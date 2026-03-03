import json

INPUT_FILE = "data/synthetic_eval_test.json"
OUTPUT_FILE = "data/eval_no_borderline.json"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

filtered = [
    obj for obj in data
    if obj.get("category") != "benign_request_borderline_response"
]

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2)

print(f"Removed {len(data) - len(filtered)} entries.")
print(f"Saved {len(filtered)} entries to {OUTPUT_FILE}.")