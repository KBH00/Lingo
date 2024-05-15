from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from request_check import pre_processing
from non_ai import *
import time

"""
request body:
{
  model: "string"
  targetWord: "string" or "integer"
  sentence: bool
  cntxt_len: "integer"
  text: "string"
  abbrebiation: bool
}

response body:
{
 suggenstion: {
       "synonym1":"string(def) or score"
        ...
       }
}
"""

prev_tokenizer = None
prev_model_name = None
prev_model = None

class synonym_recommendation():

    def __init__(self, model, targetWord, sentence, cntxt_len, text) -> None:
        self.model = model
        self.target_word = targetWord
        self.sentence = sentence
        self.cntxt_len = cntxt_len
        self.text = text
    
    def convert_dict(self, terms_list) -> dict:
        terms_dict = {}
        for term_definition in terms_list:
            term = term_definition
            if '-' in term_definition:
                term, definition = term_definition.split(' - ')
            if '('  in term:
                term = term.split('(', 1)[0]
            term = term.strip()
            definition = definition.strip()
            terms_dict[term] = definition
        return terms_dict
    
    def with_score(self, terms_list) -> dict:
        terms_dict = {}
        #for term in terms_list:
            

    def extract_medical_terms(self, text) -> list:
        """
        Extract medical terms from a given text block, assuming each term is listed in a numbered format.
        
        Args:
        - text (str): A block of text containing medical terms listed in a numbered format.
        
        Returns:
        - list: A list of extracted medical terms.
        """
        extracted_terms = []
        lines = text.split('\n')
        
        for line in lines:
            if line.strip().startswith(tuple(str(i) for i in range(1, 6))):
                term = line.split('. ', 1)[-1]
                if '('  in term:
                    term = term.split('(', 1)[0]
                extracted_terms.append(term)
        #print(extracted_terms)
        return extracted_terms
    
    def post_processing(self) -> dict:
        global prev_model_name, prev_model, prev_tokenizer

        # if prev_model_name == self.model:
        #     terms = prev_model.inference()
        
        # else:

        base_tokenizer, base_model, class_model = pre_processing(model=self.model, targetWord=self.target_word, cntxt_len=self.cntxt_len, 
                            sentence=self.sentence, text=self.text, prev_model_str=prev_model_name, prev_model=prev_model, base_tokenizer=prev_tokenizer).model_check()
        
        prev_model=base_model
        prev_model_name=self.model
        prev_tokenizer=base_tokenizer

        st = time.time()
        terms = class_model.inference()
        et = time.time()

        print('\n-------')
        print(et-st)
        print('-------')

        if self.model == 'llama2':
            terms = self.extract_medical_terms(terms)
            #terms = self.convert_dict(terms_list)

        return terms


import argparse

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
       "synonym1":"string(def) or score"
        ...
       }
}
"""

def get_args_parser():
    parser = argparse.ArgumentParser('Set Synonym recommendation', add_help=False)
    parser.add_argument('--model', default='llama2', type=str, help="name of model to inference")
    parser.add_argument('--target_word', default='', type=str, help='word to find synonym')

    parser.add_argument('--sentence', default=True, type=bool, help='Unit for context consideration')
    parser.add_argument('--context_len', default=1, type=int, help='Number of unit')

    parser.add_argument('--text', default='', type=str, help='Context of word')
    parser.add_argument('--abbrv', default=False, type=bool, help='Turn on or off to change abbreviation')
     
# def main():
#     synonym_recommend = synonym_recommendation(model="llama2", targetWord="CT", sentence=True, cntxt_len=2, text="The radiological evaluation of patients with acute spinal trauma has always been a challenging problem. Multiple radiological procedures are often necessary for \
#                                                complete evaluation of the extent of spinal injury. CT provides an ideal modality whereby accurate assessment of displacement of bony fragments as well as associated spinal cord and nerve root injury can easily be performed, \
#                                                eliminating the need for difficult radiological procedures.")
#     print(synonym_recommend.post_processing())
# while True:
#     a = synonym_recommendation(model="llama2", targetWord="BP", sentence=True, cntxt_len=2, text="")
#     print(a.post_processing())

# a = synonym_recommendation(model="llama2", targetWord="CT", sentence=False, cntxt_len=0, text="The radiological evaluation of patients with acute spinal trauma has always been a challenging problem. Multiple radiological procedures are often necessary for complete evaluation of the extent of spinal injury. CT provides an ideal modality whereby accurate assessment of displacement of bony fragments as well as associated spinal cord and nerve root injury can easily be performed, eliminating the need for difficult radiological procedures.")
# print(a.post_processing())

a = synonym_recommendation(model="llama2", targetWord="CT", sentence=True, cntxt_len=2, text="The radiological evaluation of patients with acute spinal trauma has always been a challenging problem. Multiple radiological procedures are often necessary for complete evaluation of the extent of spinal injury. CT provides an ideal modality whereby accurate assessment of displacement of bony fragments as well as associated spinal cord and nerve root injury can easily be performed, eliminating the need for difficult radiological procedures.")
print(a.post_processing())

a = synonym_recommendation(model="llama2", targetWord="MRI", sentence=True, cntxt_len=0, text="The radiological evaluation of patients with acute spinal trauma has always been a challenging problem. Multiple radiological procedures are often necessary for complete evaluation of the extent of spinal injury. CT provides an ideal modality whereby accurate assessment of displacement of bony fragments as well as associated spinal cord and nerve root injury can easily be performed, eliminating the need for difficult radiological procedures.")
print(a.post_processing())

