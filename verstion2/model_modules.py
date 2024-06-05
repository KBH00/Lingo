from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from abc import ABC, abstractmethod
import torch
import os
from openai import OpenAI

MODEL_SYNONYM_SERVICE_MAPPING_NAME = {
    'huggingface': 'HuggingFaceSynonymModel',
    'llama2-7b': 'Llama2Model',
    'openai': 'OpenAISynonymModel',
    # Add other model services here
}

class AbstractModel(ABC):
    @abstractmethod
    def suggest_synonyms(self, word: str) -> dict:
        pass

class HuggingFaceSynonymModel(AbstractModel):
    def __init__(self, model_name: str = "facebook/nllb-200-distilled-1.3B"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def suggest_synonyms(self, word: str) -> dict:
        inputs = self.tokenizer(word, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        synonyms = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return {"synonyms": synonyms}

class Llama2Model(AbstractModel):
    def __init__(self, model_name: str = "llama2-7b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def suggest_synonyms(self, word: str) -> dict:
        prompt = f"Generate synonyms for the word: {word}"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        synonyms_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        synonyms = [syn.strip() for syn in synonyms_text.split(",")]
        return {"synonyms": synonyms}

class OpenAISynonymModel(AbstractModel):
    def __init__(self):
        self.auth_key = os.getenv("OPENAI_AUTH_KEY")
        self.client = OpenAI(api_key=self.auth_key)

    def suggest_synonyms(self, word: str) -> dict:
        prompt = f"Suggest synonyms for the word: {word}"

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            model="gpt-3.5-turbo",
        )
        synonyms_text = chat_completion.choices[0].message.content
        synonyms = [syn.strip() for syn in synonyms_text.split(", ")]  # Assuming response is comma-separated
        return {"synonyms": synonyms}
