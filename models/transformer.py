import tensorflow as tf
import numpy as np

def positional_encoding(length, depth):
    depth = depth // 2

    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth

    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates

    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)],
        axis=-1
    )

    return tf.cast(pos_encoding, dtype=tf.float32)

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

def masked_accuracy(label, pred):
    pred = tf.argmax(pred, axis=2)
    label = tf.cast(label, pred.dtype)
    match = label == pred

    mask = label != 0
    match = match & mask

    match = tf.cast(match, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    return tf.reduce_sum(match) / tf.reduce_sum(mask)

@tf.keras.utils.register_keras_serializable()
class Embedding(tf.keras.layers.Layer):
    def __init__(self, *, max_length, vocab_size, d_model, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.max_length = max_length
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate

        self.embedding = tf.keras.layers.Embedding(
            vocab_size, d_model, mask_zero=True
        )
        self.pos_encoding = positional_encoding(max_length, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return self.dropout(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "max_length": self.max_length,
            "d_model": self.d_model,
            "vocab_size": self.vocab_size,
            "dropout_rate": self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class Attention(tf.keras.layers.Layer):
    def __init__(self,*, num_heads, key_dim, dropout_rate=0.1, use_causal_mask=False, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.use_causal_mask = use_causal_mask
        self.dropout_rate = dropout_rate

        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, 
            key_dim=key_dim,
            dropout=dropout_rate
        )
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, query, key, value):
        # Keras MultiHeadAttention handles masks automatically when use_causal_mask=True
        attn_output = self.mha(
            query=query,
            key=key,
            value=value,
            use_causal_mask=self.use_causal_mask
        )
        x = self.add([query, attn_output])
        return self.layernorm(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "dropout_rate": self.dropout_rate,
            "use_causal_mask": self.use_causal_mask,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate),
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        return self.layer_norm(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.self_attention = Attention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout_rate=dropout_rate,
            use_causal_mask=True
        )
        self.ffn = FeedForward(d_model, dff, dropout_rate)

    def call(self, x):
        x = self.self_attention(query=x, key=x, value=x)
        return self.ffn(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
        })
        return config

@tf.keras.utils.register_keras_serializable()
class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.enc_layers = [
            DecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dff=dff,
                dropout_rate=dropout_rate,
            )
            for _ in range(num_layers)
        ]

    def call(self, x):
        for layer in self.enc_layers:
            x = layer(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
        })
        return config
    
@tf.keras.utils.register_keras_serializable()
class Model(tf.keras.Model):
    def __init__(
        self,
        *,
        num_layers,
        d_model,
        num_heads,
        dff,
        max_length,
        vocab_size,
        dropout_rate=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.dropout_rate = dropout_rate

        self.dec_embedding = Embedding(
            vocab_size=self.vocab_size,
            d_model=d_model,
            max_length=max_length,
            dropout_rate=dropout_rate,
        )

        self.decoder = Decoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
        )

        self.out = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs):
        dec_embed = self.dec_embedding(inputs)
        dec_out = self.decoder(dec_embed)

        return self.out(dec_out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "max_length": self.max_length,
            "vocab_size": self.vocab_size,
            "dropout_rate": self.dropout_rate,
        })
        return config