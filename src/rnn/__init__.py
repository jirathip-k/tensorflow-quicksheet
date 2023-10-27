import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

from .rnn import RNN
from typing import cast
import argparse


def parse_state_sizes(string):
    return [int(item) for item in string.split(",")]


def build_model(variant, batch_size, seq_len, vocab_size, embed_size, state_sizes):
    print("TensorFlow version:", tf.__version__)
    print("GPU: ", tf.config.list_physical_devices("GPU"))
    x = tf.random.uniform(
        (batch_size, seq_len), minval=0, maxval=vocab_size, dtype=tf.int32
    )
    # x = tf.keras.Input([seq_len], dtype="int64")
    input_shape = cast(tf.Tensor, x).shape
    print(f"Input shape: {input_shape}")

    rnn = RNN(
        variant=variant,
        vocab_size=vocab_size,
        embed_size=embed_size,
        state_sizes=state_sizes,
    )
    rnn_output_shape = cast(tf.Tensor, rnn(x)).shape
    print(f"RNN output shape: {rnn_output_shape}")

    for layer in rnn.layers:
        print(f"Layer Name: {layer.name}")

        # Fetch weights and their names
        weight_tensors = layer.weights
        weight_names = [w.name for w in layer.trainable_weights]

        for name, weight in zip(weight_names, weight_tensors):
            weight_type = name.split("/")[-1].split(":")[0]
            print(f"{weight_type}: {weight.shape}")


def main():
    parser = argparse.ArgumentParser(description="RNN Example")

    parser.add_argument(
        "--variant", type=str, default="SimpleRNN", help="Variant of RNN"
    )
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
    build_model(
        args.variant,
        args.batch_size,
        args.seq_len,
        args.vocab_size,
        args.embed_size,
        parse_state_sizes(args.state_sizes),
    )


if __name__ == "__main__":
    main()
