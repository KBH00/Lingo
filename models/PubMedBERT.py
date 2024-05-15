from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
from scipy.spatial.distance import cosine

# tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
# model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

class med_BERT_embedding():

    def __init__(self, tokenizer, model, input_word, word_list, top_k=5):
        self.tokenizer = tokenizer
        self.model = model
        self.input_word = input_word
        self.word_list = word_list
        self.top_k = top_k

    def get_word_embedding(self, word):
        inputs = self.tokenizer(word, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(1)  
        embeddings = embeddings.squeeze().detach().numpy()  
        return embeddings

    def inference(self):
        input_embedding = self.get_word_embedding(f"This is about {self.input_word}.")

        similarities = []
        for word in self.word_list:
            word_embedding = self.get_word_embedding(f"This is about {word}.")
            similarity = 1 - cosine(input_embedding, word_embedding)
            similarities.append((word, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_k_similar_words = similarities[:self.top_k]
        top_k_dict = {word:similarity for word, similarity in top_k_similar_words}
        return top_k_dict

# input_word = "cancer"
# word_list = ["disease", "computed tomography", "magnetic resonance imaging", "ultrasound", "X-ray", "radiography"]
# top_k_similar_words = find_top_k_similar_words(input_word, word_list, top_k=3)

# print("Top k similar words:")
# for word, similarity in top_k_similar_words:
#     print(f"{word}: {similarity}")
