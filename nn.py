import keras
from keras import layers


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        # Multi-head attention (Causal mask is applied here later)
        attn_output = self.att(inputs, inputs, use_causal_mask=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = keras.ops.shape(x)[-1]
        positions = keras.ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def build_tiny_transformer(
    vocab_size, max_len, embed_dim, num_heads, ff_dim, num_layers
):
    keras.mixed_precision.set_global_policy("mixed_float16")
    inputs = layers.Input(shape=(max_len,), dtype="int32")

    # Embedding Layer
    embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
    x = embedding_layer(inputs)

    # Stack Transformer Blocks
    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim)(x)

    # Output Head
    outputs = layers.Dense(vocab_size, activation=None)(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def compile_model(model):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, ignore_class=0
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=1e-4, weight_decay=0.01, clipnorm=1.0
    )
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model
