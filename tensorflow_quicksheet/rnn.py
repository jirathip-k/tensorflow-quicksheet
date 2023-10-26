import tensorflow as tf
from typing import cast


class RNN(tf.keras.Model):
    def __init__(
        self, vocab_size: int, embed_size: int, state_sizes: list[int]
    ) -> None:
        super().__init__()
        self.embeddings_table = tf.keras.layers.Embedding(
            vocab_size, embed_size, mask_zero=True
        )

        self.rnns = [
            tf.keras.layers.SimpleRNN(state_size, return_sequences=True)
            for state_size in state_sizes
        ]

    def call(self, x: tf.Tensor) -> tf.Tensor:
        x = cast(tf.Tensor, self.embeddings_table(x))

        for rnn in self.rnns:
            x = cast(tf.Tensor, rnn(x))

        return x
