import tensorflow as tf


class Tokeniser:
    def __init__(self):
        self.vocab = set()
        self.vocab_size = 0
        self.word2id = {}
        self.id2word = {}

    def tokenise(self, sentences):
        tokens = [word for sentence in sentences for word in sentence.split()]
        self.vocab = set(tokens)
        self.vocab_size = len(self.vocab)
        self.word2id = {word: i for i, word in enumerate(self.vocab)}
        self.id2word = {i: word for word, i in self.word2id.items()}

        return self


def create_dataset(sentences, word2id, window_size):
    all_targets = []
    all_contexts = []

    for sentence in sentences:
        tokens = sentence.split()

        if len(tokens) < 2 * window_size + 1:
            continue

        pairs = []
        for i in range(window_size, len(tokens) - window_size):
            target = word2id[tokens[i]]
            context_candidates = (
                tokens[i - window_size : i] + tokens[i + 1 : i + window_size + 1]
            )
            for context in context_candidates:
                pairs.append((target, word2id[context]))

        targets, contexts_list = zip(*pairs)
        all_targets.extend(targets)
        all_contexts.extend(contexts_list)

    targets_tensor = tf.convert_to_tensor(all_targets, dtype=tf.int32)
    contexts_tensor = tf.convert_to_tensor(all_contexts, dtype=tf.int32)

    return targets_tensor, contexts_tensor
