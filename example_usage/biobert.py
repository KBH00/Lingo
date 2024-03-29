from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from scipy.spatial.distance import cosine

tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")


def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(1)  
    embeddings = embeddings.squeeze().detach().numpy()  
    return embeddings

def find_top_k_similar_words(input_word, word_list, top_k=5):
    if input_word in tokenizer.get_vocab():
        print(f"'{input_word}' is in the model's vocabulary. Finding similar words...")
        input_embedding = get_word_embedding(f"This is about {input_word}.")

        similarities = []
        for word in word_list:
            word_embedding = get_word_embedding(f"This is about {word}.")
            similarity = 1 - cosine(input_embedding, word_embedding)
            similarities.append((word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    else:
        print(f"'{input_word}' is not in the model's vocabulary. Comparing with given word list...")
        input_embedding = get_word_embedding(input_word)
        similarities = []
        for word in word_list:
            word_embedding = get_word_embedding(word)
            similarity = 1 - cosine(input_embedding, word_embedding) 
            similarities.append((word, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# Example usage
input_word = "MRI"
word_list = ["computed tomography", "magnetic resonance imaging", "ultrasound", "X-ray", "radiography"]
top_k_similar_words = find_top_k_similar_words(input_word, word_list, top_k=3)
print("Top k similar words:")
for word, similarity in top_k_similar_words:
    print(f"{word}: {similarity}")
