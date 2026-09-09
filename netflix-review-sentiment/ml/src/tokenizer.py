import re

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    tokens = text.split()
    return tokens

review = "This movie was absolutely amazing! I loved every minute of it."

tokens = tokenize(review)

print("Original:")
print(review)
print("\nTokens:")
print(tokens)