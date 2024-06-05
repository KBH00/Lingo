import os
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from abc import ABC, abstractmethod
import torch
from openai import OpenAI
from dotenv import load_dotenv
from typing import Dict, Any

# Ensure you have your dotenv file loaded
load_dotenv()

MODEL_SYNONYM_SERVICE_MAPPING_NAME = {
    'huggingface': 'HuggingFaceSynonymModel',
    'llama2-7b': 'Llama2Model',
    'openai': 'OpenAISynonymModel',
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
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        synonyms = [syn.strip() for syn in generated_text.split(',')]
        return {"synonyms": synonyms}

class Llama2Model(AbstractModel):
    def __init__(self, model_name: str = "llama2-7b"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def suggest_synonyms(self, word: str) -> dict:
        prompt = f"Generate synonyms for the word: {word}"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        synonyms = [syn.strip() for syn in generated_text.split(',')]
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
        synonyms = [syn.strip() for syn in synonyms_text.split(", ")]
        return {"synonyms": synonyms}

def get_class(modules, class_name):
    return getattr(modules, class_name, None)

class SynonymAPIManager:
    _api_mapping: Dict[str, str] = {}  # Add your API services if any

    def __init__(self, service: str, **kwargs: Dict[str, Any]) -> None:
        self.kwargs = kwargs
        self.api: AbstractModel = self.initialize_api(service)

    def initialize_api(self, service: str) -> AbstractModel:
        if service in self._api_mapping:
            api_class_name = self._api_mapping[service]
            api_class = get_class(model_modules, api_class_name)
            if not api_class:
                raise ValueError(f"No API class found for {api_class_name}")
            return api_class(**self.kwargs)
        else:
            raise ValueError(f"Unsupported service: {service}")

    def suggest_synonyms(self, word: str, **kwargs) -> dict:
        result = self.api.suggest_synonyms(word, **kwargs)
        return result

    def change_service(self, service: str) -> None:
        self.api = self.initialize_api(service)

class SynonymModelManager:
    _model_mapping: Dict[str, str] = MODEL_SYNONYM_SERVICE_MAPPING_NAME

    def __init__(self, service: str, model_name: str = None, **kwargs: Dict[str, Any]) -> None:
        self.kwargs = kwargs
        self.service = service
        self.model_name = model_name
        self.model: AbstractModel = self.initialize_model(service, model_name)

    def initialize_model(self, service: str, model_name: str = None) -> AbstractModel:
        if service in self._model_mapping:
            model_class_name = self._model_mapping[service]
            model_class = get_class(__name__, model_class_name)
            if not model_class:
                raise ValueError(f"No model class found for {model_class_name}")
            if model_name:
                return model_class(model_name=model_name, **self.kwargs)
            return model_class(**self.kwargs)
        else:
            raise ValueError(f"Unsupported translation service: {service}")

    def suggest_synonyms(self, word: str, **kwargs) -> dict:
        result = self.model.suggest_synonyms(word, **kwargs)
        return result

    def change_service(self, service: str, model_name: str = None) -> None:
        self.service = service
        self.model_name = model_name
        self.model = self.initialize_model(service, model_name)

class SynonymSuggester:
    def __init__(self):
        load_dotenv()
        self.api_manager: SynonymAPIManager = SynonymAPIManager("thesaurus")
        self.model_manager: SynonymModelManager = SynonymModelManager("huggingface")

    def suggest_synonyms(self, word: str, service: str = "huggingface", model_name: str = None, **kwargs: Dict[str, Any]) -> dict:
        if service in MODEL_SYNONYM_SERVICE_MAPPING_NAME:
            manager = self.model_manager
            manager.change_service(service, model_name)
        else:
            raise ValueError("Unsupported synonym suggestion service.")

        result = manager.suggest_synonyms(word, **kwargs)
        return result

# Example usage
synonym_suggester = SynonymSuggester()

# Suggest synonyms using a specific service and model
try:
    synonyms_huggingface = synonym_suggester.suggest_synonyms("happy", service="huggingface", model_name="facebook/nllb-200-distilled-1.3B")
    print("HuggingFace Synonyms:", synonyms_huggingface)

    synonyms_llama2 = synonym_suggester.suggest_synonyms("happy", service="huggingface", model_name="llama2-7b")
    print("Llama2 Synonyms:", synonyms_llama2)

    synonyms_openai = synonym_suggester.suggest_synonyms("happy", service="openai")
    print("OpenAI Synonyms:", synonyms_openai)

except ValueError as e:
    print(e)
