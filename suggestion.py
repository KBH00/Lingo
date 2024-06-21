from model_load import ModelLoader
from context import get_context
from abbreviation.convert_abbrv import *
from exception import (
    ServiceNotFoundException,
    OutputFormatNotValidException,
)
#from dotenv import load_dotenv

class synonym_suggestion():

    def __init__(self, model='llama2_7b') -> None:

        """
        model : string, choose model to suggest synonym

        target_word : integer, index of target word

        sentence : bool, choose sentence-by-sentence or word-by-word

        cntxt_len : integer, number of units to check context

        text : string, paragraph including target word

        abbreviation : bool, whether or not to abbreviation conversion
        """

        #load_dotenv()
        #self.model = model
        self.model = ModelLoader(model).model_return()
    
    def suggestion(self, target_word, sentence=True, 
                   cntxt_len=0, text="", abbreviation=False) -> list:
        
        # self.target_word = target_word
        # self.sentence = sentence
        # self.cntxt_len = cntxt_len
        # self.text = text
        # self.abbreviation = abbreviation
        
        text = get_context(target_word=target_word, context=text, sentence=sentence, cntxt_len=cntxt_len) if cntxt_len != 0 else text

        if abbreviation:
           target_word= replace_abbreviations(target_word)
           return [target_word] + self.model.suggestion(target_word, text, cntxt_len)
        
        else:
            return self.model.suggestion(target_word, text, cntxt_len)

suggestor = synonym_suggestion("llama2_13b")
term = suggestor.suggestion("CT", True, 0, "what you wanna do my man? i don't need a baby right now, i just want to get a sleep.")

print(term)

