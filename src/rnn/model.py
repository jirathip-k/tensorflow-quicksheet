import tensorflow as tf
from typing import cast, List, Union, Dict

RNN_VARIANTS: Dict[
    str, Union[tf.keras.layers.SimpleRNN, tf.keras.layers.LSTM, tf.keras.layers.GRU]
] = {
    "SimpleRNN": tf.keras.layers.SimpleRNN,
    "LSTM": tf.keras.layers.LSTM,
    "GRU": tf.keras.layers.GRU,
}


class RNN(tf.keras.Model):
    def __init__(
        self,
        variant: str,
        vocab_size: int,
        embed_size: int,
        state_sizes: list[int],
    ) -> None:
        super().__init__()
        print(
            f"Creating model with vocab_size: {vocab_size} | embed_size: {embed_size} | state_sizes: {state_sizes}"
        )

        self.embeddings_table = tf.keras.layers.Embedding(
            vocab_size, embed_size, mask_zero=True, name="embeddings_table"
        )

        # Check if variant is valid
        if variant not in RNN_VARIANTS:
            raise ValueError(
                f"Unsupported RNN variant: {variant}. Supported variants are {list(RNN_VARIANTS.keys())}."
            )

        rnn_layer = RNN_VARIANTS[variant]

        self.rnns = [
            rnn_layer(state_size, return_sequences=True, name=f"{variant}_{i}")
            for i, state_size in enumerate(state_sizes)
        ]

        self.ffwd = tf.keras.layers.Dense(vocab_size, name="Logits")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: (batch, seq) -> (batch, seq, embed_size)
        x = cast(tf.Tensor, self.embeddings_table(x))

        # x: (batch, seq, embed_size) -> (batch, seq, state_size)
        for rnn in self.rnns:
            x = rnn(x)  # type: ignore

            print(f"Output of {rnn.name} shape: {x.shape}")  # type: ignore

        # x: (batch, seq, state_size) -> (batch, state_size)
        x = x[:, -1, :]  # type: ignore

        # x: (batch, state_size) -> (batch, vocab_size)
        x = self.ffwd(x)  # type: ignore
        return x
