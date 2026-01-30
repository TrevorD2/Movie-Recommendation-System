import tensorflow as tf
from pathlib import Path

import metrics
import transformer
from _config_loader import load_config
from dataset import Dataset

cfg = load_config()

# Data config
TRAIN_PATH = Path(cfg["data"]["train_path"])
VAL_PATH = Path(cfg["data"]["val_path"])
MAX_SEQ_LEN = cfg["data"]["max_seq_len"]
VOCAB_SIZE = cfg["data"]["vocab_size"] + 1 # Add 1 for padding token

# Training hyperparams
BATCH_SIZE = cfg["training"]["batch_size"]
EPOCHS = cfg["training"]["epochs"]
LEARNING_RATE = float(cfg["training"]["learning_rate"])

# Evaluation
CFG_METRICS = cfg["evaluation"]["metrics"]
EVAL_K = cfg["evaluation"]["eval_k"]

# Model hyperparams
D_MODEL = cfg["model"]["d_model"]
DFF = cfg["model"]["dff"]
NUM_LAYERS = cfg["model"]["num_layers"]
NUM_HEADS = cfg["model"]["num_heads"]
DROPOUT = cfg["model"]["dropout"]

OUT_PATH = Path(cfg["output"]["out_model_path"])
OUT_DIR = OUT_PATH.parent
OUT_DIR.mkdir(parents=True, exist_ok=True)

def train() -> None:

    train = Dataset(TRAIN_PATH, MAX_SEQ_LEN, BATCH_SIZE).get_dataset()
    val = Dataset(VAL_PATH, MAX_SEQ_LEN, BATCH_SIZE, shuffle=False).get_dataset()

    model = transformer.Model(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        dff=DFF,
        max_length=MAX_SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        dropout_rate=DROPOUT
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        clipnorm=1.0
    )

    eval_metrics = metrics.get_eval_metrics(cfg)

    model.compile(
        loss=metrics.masked_loss,
        optimizer=optimizer,
        metrics=eval_metrics
    )

    model.fit(train, epochs=EPOCHS, validation_data=val)


    model.save(OUT_PATH)


if __name__ == "__main__":
    train()