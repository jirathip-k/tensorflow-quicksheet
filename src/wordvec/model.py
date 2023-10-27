import tensorflow as tf


class SkipGram(tf.keras.Model):
    def __init__(self, vocab_size: int, embed_size: int):
        super().__init__()

        self.embeddings_table = tf.keras.layers.Embedding(
            vocab_size, embed_size, name="embeddings_table"
        )

    def call(self, target):
        # target: (batch,) -> (batch, embed_size)
        target = self.embeddings_table(target)

        logits = tf.matmul(target, self.embeddings_table.embeddings, transpose_b=True)
        return logits
