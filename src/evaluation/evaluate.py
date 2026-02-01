import tensorflow as tf
from pathlib import Path

import model.transformer
import evaluation.metrics as metrics
from dataset.dataset import Dataset
from scripts._config_loader import load_config

cfg = load_config()

MODEL_PATH = Path(cfg["output"]["out_model_path"])
TEST_PATH = Path(cfg["data"]["test_path"])
MAX_SEQ_LEN = cfg["data"]["max_seq_len"]
VOCAB_SIZE = cfg["data"]["vocab_size"] + 1 # Add 1 for padding token
BATCH_SIZE = cfg["training"]["batch_size"]

def evaluate():
    test = Dataset(TEST_PATH, MAX_SEQ_LEN, BATCH_SIZE, shuffle=False).get_dataset()

    model = tf.keras.models.load_model(MODEL_PATH, custom_objects={
        "mrr": metrics.mrr,
        "hit": metrics.hit,
        "ndcg": metrics.ndcg,
        "masked_loss": metrics.masked_loss
    })

    model.evaluate(test)

if __name__ == '__main__':
    evaluate()