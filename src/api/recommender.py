import tensorflow as tf

from scripts._config_loader import load_config

import model.transformer

cfg = load_config()

DEFAULT_PATH = cfg["output"]["out_model_path"]
MAX_SEQ_LEN = cfg["data"]["max_seq_len"]

class Recommender:
    def __init__(self, model_path=DEFAULT_PATH):
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def recommend(self, interaction_history: list[int], k: int) -> list[int]:
        self._truncate(interaction_history, MAX_SEQ_LEN)

        history_size = len(interaction_history)

        self._pad(interaction_history, MAX_SEQ_LEN)

        model_input = tf.expand_dims(tf.constant(interaction_history), axis=0)

        logits = self.model(model_input, training=False)

        predictions = logits[0, history_size, :]

        top_k = tf.argsort(predictions, direction="DESCENDING")[:k]

        return top_k.numpy().tolist()

    def _pad(self, sequence, length):
        while len(sequence) < length:
            sequence.append(0)

    def _truncate(self, sequence, length):
        while len(sequence) > length:
            sequence.pop(0)


if __name__ == '__main__':
    rec = Recommender()

    print(rec.recommend([1, 4, 3, 1, 4], 10))