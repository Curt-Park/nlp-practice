"""Test the embeddings."""
from pathlib import Path

from gensim.models.keyedvectors import KeyedVectors

model = KeyedVectors.load_word2vec_format(Path.cwd() / "word_vectors.txt", binary=False)

print("show")
print(model.most_similar(positive="show"))
print("----------------")
print("slave")
print(model.most_similar(positive="slave"))
