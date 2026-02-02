import pandas as pd
from pathlib import Path

from scripts._config_loader import load_config


cfg = load_config()

RAW_PATH = Path(cfg["data"]["raw_path"])

OUT_DIR = Path(cfg["data"]["train_path"]).parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = Path(cfg["data"]["train_path"])
VAL_PATH = Path(cfg["data"]["val_path"])
TEST_PATH = Path(cfg["data"]["test_path"])

MIN_SEQ_LEN = cfg["data"]["min_seq_len"]
TRAIN_FRAC = cfg["data"]["train_frac"]
VAL_FRAC = cfg["data"]["val_frac"]

def load_ratings(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    return df

def build_sequences(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["user_id", "timestamp"])

    seqs = (
        df.groupby("user_id")["item_id"]
        .apply(list)
        .reset_index(name="item_ids")
    )

    seqs = seqs[seqs["item_ids"].apply(len) >= MIN_SEQ_LEN]

    return seqs

def split_sequences(df: pd.DataFrame):
    train, val, test = [], [], []

    for _, row in df.iterrows():
        items = row['item_ids']
        num_items = len(items)

        train_end = int(num_items * TRAIN_FRAC)
        val_end = int(num_items * (TRAIN_FRAC + VAL_FRAC))

        train.append(items[:train_end])
        val.append(items[:val_end])
        test.append(items)

    return train, val, test

def save_data(seqs, path: Path):
    pd.DataFrame({"item_ids": seqs}).to_parquet(path, index=False)


def main(head=False, num_samples=5):
    df = load_ratings(RAW_PATH)
    seq_df = build_sequences(df)

    train, val, test = split_sequences(seq_df)

    save_data(train, TRAIN_PATH)
    save_data(val, VAL_PATH)
    save_data(test, TEST_PATH)

    print(f"Saved {len(train)} train sequences")
    print(f"Saved {len(val)} val sequences")
    print(f"Saved {len(test)} test sequences")

    if head:
        print(f"FIRST {num_samples} SAMPLES:")

        for i in range(num_samples):
            print(f"SAMPLE {i}:")
            print(train[i])


if __name__ == "__main__":
    main(head=True)