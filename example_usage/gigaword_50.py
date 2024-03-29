from gensim.models import KeyedVectors
from gensim.downloader import load

def find_similar_words(input_word, top_k=5):
    model = load('glove-wiki-gigaword-50')
    similar_words = model.most_similar(input_word, topn=top_k)

    return similar_words

input_word = "CT"  # You can change this to any word of your choice
top_k = 5  # Number of top similar words to find
similar_words = find_similar_words(input_word, top_k)

print(f"Top {top_k} words similar to '{input_word}':")
for word, similarity in similar_words:
    print(f"{word}: {similarity}")
