import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from .model import SkipGram
from .tokeniser import Tokeniser, create_dataset
import tensorflow as tf


def main():
    sentences = ["the mouse runs away from the cat", "the cat chases the mouse"]
    window_size = 2

    tokeniser = Tokeniser().tokenise(sentences)

    targets, contexts = create_dataset(sentences, tokeniser.word2id, window_size=2)
    skipgram = SkipGram(vocab_size=tokeniser.vocab_size, embed_size=10)

    loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    for _ in range(10):
        with tf.GradientTape() as tape:
            logits = skipgram(targets)

            loss = loss_func(contexts, logits)
            print(f"{loss}")

        grad = tape.gradient(loss, skipgram.trainable_variables)
        optimizer.apply_gradients(zip(grad, skipgram.trainable_variables))  # type: ignore
