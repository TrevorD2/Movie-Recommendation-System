import tensorflow as tf
import pandas as pd
from pathlib import Path

class Dataset:
    def __init__(
            self,
            path: Path,
            max_seq_len: int,
            batch_size: int,
            shuffle: bool = True
    ):
        self.path = path
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle

    def _load_sequences(self):
        df = pd.read_parquet(self.path)
        return df["item_ids"].tolist()
    
    def _truncate(self, seq):
        # Keep most recent interactions
        if len(seq) > self.max_seq_len + 1:
            seq = seq[-(self.max_seq_len + 1):]
        return seq
    
    def _split_input_target(self, seq):
        seq = self._truncate(seq)
        
        input_seq = seq[:-1] # Remove last interaction
        target_seq = seq[1:] # Remove first interaction

        return input_seq, target_seq
    

    def get_dataset(self) -> tf.data.Dataset:
        sequences = self._load_sequences()

        def sequence_generator():
            for seq in sequences:
                x, y = self._split_input_target(seq)
                yield x, y

        output_signature = (
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        )

        ds = tf.data.Dataset.from_generator(sequence_generator, output_signature=output_signature)

        if self.shuffle:
            ds = ds.shuffle(buffer_size=10_000)

        ds = ds.padded_batch(
            self.batch_size,
            padded_shapes=([self.max_seq_len], [self.max_seq_len]),
            padding_values=(0, 0),
            drop_remainder=True
        )

        return ds.prefetch(tf.data.AUTOTUNE)