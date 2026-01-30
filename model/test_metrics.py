import unittest
import numpy as np

from metrics import *

from _config_loader import load_config

cfg = load_config()
VOCAB_SIZE = cfg["data"]["vocab_size"]
MAX_SEQ_LEN = cfg["data"]["max_seq_len"]
EVAL_K = cfg["evaluation"]["eval_k"]

def pad(seq, target_len) -> None:
    while len(seq) < target_len:
        seq.append(0)

class TestMetrics(unittest.TestCase):
    def setUp(self):
        label = [3, 2, 4, 5, 6]


        # Only care about index 4, so place dist. of zeros in all other locations
        zeros_distribution = [0 for _ in range(VOCAB_SIZE)]
        pred = [zeros_distribution for _ in range(MAX_SEQ_LEN)]

        last_indx_distribution = [0 for _ in range(VOCAB_SIZE)]
        last_indx_distribution[6] = 0.5 # Actual movie probability (3rd highest likelihood)

        # Dummy probabilities
        last_indx_distribution[8] = 0.3
        last_indx_distribution[2] = 0.4
        last_indx_distribution[9] = 0.6
        last_indx_distribution[3] = 0.7

        # Pred distribution top 5:
        # MOVIES:  3  9 *6* 2  8
        # PROBS:  .7 .6 .5 .4 .3

        pred[4] = last_indx_distribution

        self.label = tf.expand_dims(tf.constant(label), axis=0)
        self.pred = tf.expand_dims(tf.constant(pred), axis=0)

    def test_mrr(self):
        expected = np.float32(1 / 3)

        actual = mrr(self.label, self.pred).numpy()

        self.assertEqual(expected, actual)

    def test_ndcg(self):
        expected = 0.5

        actual = ndcg(self.label, self.pred)

        self.assertEqual(expected, actual)

    def test_hit(self):
        expected = 1

        actual = hit(self.label, self.pred)

        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()