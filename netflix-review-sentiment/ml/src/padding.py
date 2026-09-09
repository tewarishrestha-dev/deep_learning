from tokenizer import tokenize
from vocabulary import Vocabulary


def pad_sequence(sequence, max_length, pad_value=0):

    if len(sequence) > max_length:
        return sequence[:max_length]

    return sequence + [pad_value] * (max_length - len(sequence))


reviews = [
    "This movie was absolutely amazing!",
    "I loved every minute of it.",
    "The story was terrible."
]


vocab = Vocabulary()
vocab.build(reviews)


max_length = 6

for review in reviews:

    encoded = vocab.encode(review)

    padded = pad_sequence(
        encoded,
        max_length
    )

    print("\nReview:")
    print(review)

    print("Encoded:")
    print(encoded)

    print("Padded:")
    print(padded)