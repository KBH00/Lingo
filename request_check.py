from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from models.llama2_7b import llama2_7b
from models.masking import BERT_masking
from transformers import pipeline
from transformers import AutoModel, AutoTokenizer
from models.PubMedBERT import med_BERT_embedding
import torch
import numpy as np
from scipy.spatial.distance import cosine

import logging

"""
request body:
{
  model: "string"
  targetWord: "string" or "integer"
  sentence: bool
  cntxt_len: "integer"
  text: "string"
  abbreviation: bool
}

response body:
{
 suggenstion: {
       "synonym1":"string(def)"
        ...
       }
}
"""

class pre_processing():

    def __init__(self, model, targetWord, cntxt_len, sentence, text, prev_model_str, prev_model, base_tokenizer) -> None:
        self.model = model
        self.target_word = targetWord
        self.cntxt_len = cntxt_len
        self.sentence = sentence
        self.context = text
        self.prev_model_str = prev_model_str
        self.prev_model = prev_model
        self.base_tokenizer = None
        self.curr_model = None
        self.word_list = ["disease", "computed tomography", "magnetic resonance imaging", "ultrasound", "X-ray", "radiography"]


    def get_context(self) -> str:

        if self.cntxt_len==0:
            return self.target_word
        
        sentences = self.context.split('. ')
        
        target_sentence = None
        for sent in sentences:
            if self.target_word.lower() in sent.lower():
                target_sentence = sent
                break
        
        if target_sentence is None:
            return "Target word not found in any sentence."
        
        sent_index = sentences.index(target_sentence)

        if self.sentence:
            start_index = max(0, sent_index - self.cntxt_len)
            end_index = min(len(sentences), sent_index + self.cntxt_len + 1)
            context = '. '.join(sentences[start_index:end_index])
        else:
            #words = target_sentence.split()
            words = self.context.split()
            word_index = words.index(self.target_word)
            start_index = max(0, word_index - self.cntxt_len)
            end_index = min(len(words), word_index + self.cntxt_len + 1)
            context = ' '.join(words[start_index:end_index])
#        print(context)
        return context
    
    def model_check(self):
        context=self.get_context()
        
        # print(self.prev_model_str)
        # print(self.model)
        
        if self.prev_model_str==self.model:
            if self.model == "llama2":
                return None, self.prev_model, llama2_7b(context=context, cntxt_len=self.cntxt_len, targetWord=self.target_word, model=self.prev_model)
            if self.model == "bert":
                return None, self.prev_model, BERT_masking(context=self.context, target_word=self.target_word, model=self.prev_model)
            if self.model == "medbert":
                return self.base_tokenizer, self.prev_model, med_BERT_embedding(tokenizer=self.base_tokenizer, model=model, input_word=self.target_word, word_list=self.word_list, top_k=5)
                                
        if self.model == "llama2":
            self.prev_model_str = self.model
            #callback_manager = CallbackManager([])
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
            n_gpu_layers = 40
            n_batch = 3000

            llm = LlamaCpp(
            model_path="C:/Users/kbh/Code/project2/llm/models/llama-2-7b-chat.Q2_K.gguf",
            n_gpu_layers=n_gpu_layers,
            n_batch=n_batch,
            temperature=0,  
            callback_manager=callback_manager,
            verbose=False,
            n_ctx=1800
            )

            self.curr_model = llama2_7b(context=context, cntxt_len=self.cntxt_len, targetWord=self.target_word, model=llm)
            self.prev_model = llm
        
        if self.model == "bert":
            self.prev_model_str = self.model
            logging.getLogger("transformers").setLevel(logging.WARNING)
            model_name = "bert-base-uncased"  
            llm = pipeline("fill-mask", model=model_name)

            self.prev_model = llm
            self.curr_model = BERT_masking(context=self.context, target_word=self.target_word, model=llm)
        
        if self.model == "medbert":
            self.prev_model_str = self.model

            tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
            model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

            self.base_tokenizer = tokenizer
            self.prev_model = model
            self.curr_model = med_BERT_embedding(tokenizer=tokenizer, model=model, input_word=self.target_word, word_list=self.word_list, top_k=5)
        
        return self.base_tokenizer, self.prev_model, self.curr_model





# context = "The significance of metadata in information retrieval and data management is rapidly increasing, particularly in the realms of academic research and digital content curation. To enhance usability of metadata extraction of research papers and images with person or specific animals, we propose an advanced method that surpasses existing techniques 2x accuracy by leveraging Large Language Model (LLM) and Scene Graph Generation (SGG)."
# targetWord = "academic"
# sentence = True
# cntxt_len = 1
# print(checking(model=None, targetWord=targetWord, cntxt_len=cntxt_len, sentence=sentence, text=context).get_context())


