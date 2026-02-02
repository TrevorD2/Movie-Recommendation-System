import tensorflow as tf

from scripts._config_loader import load_config

cfg = load_config()
VOCAB_SIZE = cfg["data"]["vocab_size"]
EVAL_K = cfg["evaluation"]["eval_k"]

def get_eval_metrics(cfg) -> list:
    eval_metrics = []

    if "hit" in cfg["evaluation"]["metrics"]:
        eval_metrics.append(hit)

    if "ndcg" in cfg["evaluation"]["metrics"]:
        eval_metrics.append(ndcg)

    if "mrr" in cfg["evaluation"]["metrics"]:
        eval_metrics.append(mrr)
        
    return eval_metrics

def _get_last_non_padding_token_coordinates(tensor):
    padding_mask = tf.not_equal(tensor, 0)
    numeric_mask = tf.cast(padding_mask, dtype=tf.int32)

    sequence_lengths = tf.reduce_sum(numeric_mask, axis=-1)

    last_token_indices = tf.maximum(0, sequence_lengths - 1)

    batch_indices = tf.range(tf.shape(tensor)[0])

    coordinates = tf.stack([batch_indices, last_token_indices], axis=-1)

    return coordinates

def hit(label, pred, eval_k=EVAL_K):
    coordinates = _get_last_non_padding_token_coordinates(label)

    target_movies = tf.gather_nd(label, coordinates)

    relevant_predictions = tf.gather_nd(pred, coordinates)

    top_k = tf.argsort(relevant_predictions, direction="DESCENDING")[:, :eval_k]

    hits = tf.reduce_any(tf.equal(top_k, target_movies[:, None]), axis=1)

    return tf.reduce_mean(tf.cast(hits, tf.float32))

def ndcg(label, pred, eval_k=EVAL_K):
    coordinates = _get_last_non_padding_token_coordinates(label)

    indices = tf.range(tf.shape(label)[0])
    target_movies = tf.gather_nd(label, coordinates)
    movie_coordinates = tf.stack([indices, target_movies], axis=1)

    relevant_predictions = tf.gather_nd(pred, coordinates)

    ranks = tf.argsort(tf.argsort(relevant_predictions, direction="DESCENDING")) + 1

    relevant_ranks = tf.cast(tf.gather_nd(ranks, movie_coordinates), dtype=tf.float32)

    condition = tf.less(relevant_ranks, eval_k)

    result_true = tf.math.log(2.0) / (tf.math.log(relevant_ranks + 1))

    processed = tf.where(condition, result_true, 0.0)

    return tf.reduce_mean(processed)

def mrr(label, pred):
    coordinates = _get_last_non_padding_token_coordinates(label)
    
    indices = tf.range(tf.shape(label)[0])
    target_movies = tf.gather_nd(label, coordinates)
    movie_coordinates = tf.stack([indices, target_movies], axis=1)

    relevant_predictions = tf.gather_nd(pred, coordinates)

    ranks = tf.argsort(tf.argsort(relevant_predictions, direction="DESCENDING")) + 1

    relevant_ranks = tf.gather_nd(ranks, movie_coordinates)

    reciprical_ranks = tf.math.reciprocal(tf.cast(relevant_ranks, dtype=tf.float32))

    return tf.reduce_mean(reciprical_ranks)

def masked_loss(label, pred):
    mask = label != 0
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, 
        reduction='none'
    )

    loss = loss_object(label, pred)
    mask = tf.cast(mask, dtype=loss.dtype)
    loss *= mask

    loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return loss