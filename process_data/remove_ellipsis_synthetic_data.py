import json

INPUT_FILE = "raw_dataset.json"
OUTPUT_FILE = "cleaned_dataset.json"


def contains_ellipsis(entry):
    for msg in entry.get("messages", []):
        if msg.get("role") == "assistant":
            if "..." in msg.get("content", ""):
                return True
    return False


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = [entry for entry in data if not contains_ellipsis(entry)]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Original: {len(data)}")
    print(f"Removed: {len(data) - len(cleaned)}")
    print(f"Remaining: {len(cleaned)}")


if __name__ == "__main__":
    main()