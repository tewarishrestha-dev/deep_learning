from tokenizer import tokenize

class Vocabulary:

    def __init__(self, max_vocab_size=30000):

        self.max_vocab_size = max_vocab_size

        self.word_to_index = {
            "<PAD>": 0,
            "<UNK>": 1
        }

        self.index_to_word = {
            0: "<PAD>",
            1: "<UNK>"
        }

    def build(self, texts):

        word_counts = {}

        for text in texts:

            tokens = tokenize(text)

            for token in tokens:

                word_counts[token] = (
                    word_counts.get(token, 0) + 1
                )

        # Most frequent words first
        sorted_words = sorted(
            word_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Reserve 2 positions for PAD and UNK
        sorted_words = sorted_words[
            :self.max_vocab_size - 2
        ]

        for word, _ in sorted_words:

            index = len(self.word_to_index)

            self.word_to_index[word] = index
            self.index_to_word[index] = word

    def encode(self, text):

        tokens = tokenize(text)

        return [
            self.word_to_index.get(
                token,
                self.word_to_index["<UNK>"]
            )
            for token in tokens
        ]