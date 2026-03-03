import json
from datasets import load_dataset
from tqdm import tqdm

DATASET_NAME = "HuggingFaceTB/everyday-conversations-llama3.1-2k"
SPLIT = "train_sft"
N = 750
OUTPUT_PATH = "data/everyday_conversations.json"


def reorder_message_keys(msg):
    role = msg.get("role")
    content = msg.get("content")
    return {"role": role, "content": content}


def main():
    ds = load_dataset(DATASET_NAME, split=SPLIT, streaming=True)

    out = []
    kept = 0

    pbar = tqdm(total=N, desc="Generating everydayconversations dataset")
    for row in ds:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            continue

        trimmed = messages[2:]
        trimmed = [reorder_message_keys(m) for m in trimmed if isinstance(m, dict)]

        if not trimmed:
            continue

        out.append(
            {
                "category": "benign_multi_turn",
                "label": 0,
                "messages": trimmed,
            }
        )

        pbar.update(1)

        kept += 1
        if kept >= N:
            break

    pbar.close()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(out)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()