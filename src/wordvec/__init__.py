import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from .model import SkipGram
from .tokeniser import Tokeniser, create_dataset


def main():
    sentences = ["the mouse runs away from the cat", "the cat chases the mouse"]

    tokeniser = Tokeniser().tokenise(sentences)

    targets, contexts = create_dataset(sentences, tokeniser.word2id, window_size=2)
    print(targets, contexts)
    skipgram = SkipGram(vocab_size=tokeniser.vocab_size, embed_size=10)

    out = skipgram(targets)
    print(f"Output shape {out.shape}")  # type: ignore
