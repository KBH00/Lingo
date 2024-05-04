from transformers import pipeline

class BERT_masking():
    
    def __init__(self, context, target_word, model) -> None:
        self.context= context
        self.target_word = target_word
        self.model = model

    def mask_word(self):
        words = self.context.split()
        
        masked_text = ""
        for word in words:
            if word.lower() == self.target_word.lower():
                masked_text += "[MASK] "
            else:
                masked_text += word + " "
        
        return masked_text.strip()

    def inference(self):

        text_with_example = self.mask_word()
        predictions = self.model(text_with_example)
        word_score_dict = {item['token_str']: item['score'] for item in predictions}
        sorted_word_score_dict = dict(sorted(word_score_dict.items(), key=lambda item: item[1], reverse=True))

        return sorted_word_score_dict
    
# model_name = "bert-base-uncased"  
# llm = pipeline("fill-mask", model=model_name)
# text = "This is a sample sentence with some words. We will mask a word in this text."

# print(BERT_masking(context=text, target_word="word", model=llm).inference())



