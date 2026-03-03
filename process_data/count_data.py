import json
from collections import Counter

DATA_PATH = "data/final.json"

with open(DATA_PATH, "r") as f:
    data = json.load(f)

categories_to_track = [
    "benign_request",
    "benign_single_turn",
    "benign_multi_turn",
    "harmful_request",
    "harmful_single_turn",
    "harmful_multi_turn",
]

counter = Counter()

for example in data:
    category = example.get("category", "")
    if category in categories_to_track:
        counter[category] += 1

print("Category counts:\n")
for cat in categories_to_track:
    print(f"{cat}: {counter[cat]}")