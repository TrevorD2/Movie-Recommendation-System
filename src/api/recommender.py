import tensorflow as tf
from pathlib import Path

import model.transformer
from scripts._config_loader import load_config

cfg = load_config()

DEFAULT_PATH = Path(cfg["output"]["out_model_path"])
MOVIE_METADATA_PATH = Path(cfg["data"]["movie_metadata"])
MAX_SEQ_LEN = cfg["data"]["max_seq_len"]


class Recommender:
    def __init__(self, model_path=DEFAULT_PATH, movie_metadata_path=MOVIE_METADATA_PATH):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.id_mapping = self._get_mapping(movie_metadata_path)

    def recommend(self, interaction_history: list[int], k: int) -> list[int]:
        self._truncate(interaction_history, MAX_SEQ_LEN)

        history_size = len(interaction_history)

        self._pad(interaction_history, MAX_SEQ_LEN)

        model_input = tf.expand_dims(tf.constant(interaction_history), axis=0)

        logits = self.model(model_input, training=False)

        predictions = logits[0, history_size, :]

        top_k = tf.argsort(predictions, direction="DESCENDING")[:k]

        return top_k.numpy().tolist()
    
    def translate(self, movie_ids: list[int]) -> list[str]:
        return [self.id_mapping.get(movie_id, "UNK") for movie_id in movie_ids]

    def _pad(self, sequence, length):
        while len(sequence) < length:
            sequence.append(0)

    def _truncate(self, sequence, length):
        while len(sequence) > length:
            sequence.pop(0)

    def _get_mapping(self, path: Path) -> dict[int, str]:
        mapping = {}

        with open(path, 'r') as f:
            for line in f:
                id, name, genre = line.split("::")

                mapping[int(id)] = f"{name.strip()}:{genre.strip()}"

        return mapping