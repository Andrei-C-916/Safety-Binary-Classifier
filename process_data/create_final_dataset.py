import os
import json
import random

DATA_DIR = "data"
FINAL_PATH = os.path.join(DATA_DIR, "final_dataset.json")
TRAIN_PATH = os.path.join(DATA_DIR, "train_final.json")
VAL_PATH = os.path.join(DATA_DIR, "val_final.json")

RANDOM_SEED = 42

def load_all_datasets(data_dir):
    combined = []

    for filename in os.listdir(data_dir):
        if filename.endswith(".json") and filename not in {
            "final_dataset.json",
            "train_final.json",
            "val_final.json"
        }:
            path = os.path.join(data_dir, filename)
            print(f"Loading {filename}...")

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

                if isinstance(data, list):
                    combined.extend(data)
                else:
                    print(f"Warning: {filename} is not a list. Skipping.")

    return combined

def main():
    random.seed(RANDOM_SEED)

    dataset = load_all_datasets(DATA_DIR)
    random.shuffle(dataset)

    total = len(dataset)

    train_size = int(total / 1.1)

    train_data = dataset[:train_size]
    val_data = dataset[train_size:]

    with open(FINAL_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2)

    with open(VAL_PATH, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2)

    print("\nDatasets written:")
    print(f"Total items: {total}")
    print(f"Train size: {len(train_data)}")
    print(f"Val size: {len(val_data)}")
    print(f"Val / Train ratio: {len(val_data)/len(train_data):.4f}")

if __name__ == "__main__":
    main()