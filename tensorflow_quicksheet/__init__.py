import tensorflow as tf
from tensorflow_quicksheet.rnn import RNN
from typing import cast
import argparse


def parse_state_sizes(string):
    return [int(item) for item in string.split(",")]


def main(batch_size, seq_len, vocab_size, embed_size, state_sizes):
    print("TensorFlow version:", tf.__version__)

    x = tf.random.uniform(
        (batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32
    )

    input_shape = cast(tf.Tensor, x).shape
    print(f"Input shape: {input_shape}")

    rnn = RNN(vocab_size=vocab_size, embed_size=embed_size, state_sizes=state_sizes)

    rnn_output_shape = cast(tf.Tensor, rnn(x)).shape
    print(f"RNN output shape: {rnn_output_shape}")


def wrapper():
    parser = argparse.ArgumentParser(description="RNN Example")

    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=100, help="Sequence Length")
    parser.add_argument("--vocab-size", type=int, default=100, help="Vocab size")
    parser.add_argument("--embed-size", type=int, default=10, help="Embedding size")
    parser.add_argument(
        "--state-sizes",
        type=str,
        default="32,64",
        help="State sizes as comma-separated values, e.g., '32,64'",
    )

    args = parser.parse_args()
    main(
        args.batch_size,
        args.seq_len,
        args.vocab_size,
        args.embed_size,
        parse_state_sizes(args.state_sizes),
    )


if __name__ == "__main__":
    wrapper()
