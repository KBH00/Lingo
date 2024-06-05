from dotenv import load_dotenv
from typing import Dict, Any
import api_modules
import model_modules


def get_class(modules, class_name):
    return getattr(modules, class_name, None)


class SynonymAPIManager:
    _api_mapping: Dict[str, str] = api_modules.API_SYNONYM_SERVICE_MAPPING_NAME

    def __init__(self, service: str, **kwargs: Dict[str, Any]) -> None:
        self.kwargs = kwargs
        self.api: api_modules.AbstractAPI = self.initialize_api(service)

    def initialize_api(self, service: str) -> api_modules.AbstractAPI:
        if service in self._api_mapping:
            api_class_name = self._api_mapping[service]
            api_class = get_class(api_modules, api_class_name)
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
    _model_mapping: Dict[str, str] = model_modules.MODEL_SYNONYM_SERVICE_MAPPING_NAME

    def __init__(self, service: str, model_name: str = None, **kwargs: Dict[str, Any]) -> None:
        self.kwargs = kwargs
        self.service = service
        self.model_name = model_name
        self.model: model_modules.AbstractModel = self.initialize_model(service, model_name)

    def initialize_model(self, service: str, model_name: str = None) -> model_modules.AbstractModel:
        if service in self._model_mapping:
            model_class_name = self._model_mapping[service]
            model_class = get_class(model_modules, model_class_name)
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
        if service in api_modules.API_SYNONYM_SERVICE_MAPPING_NAME:
            manager = self.api_manager
            manager.change_service(service)
        elif service in model_modules.MODEL_SYNONYM_SERVICE_MAPPING_NAME:
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
    synonyms_llama2 = synonym_suggester.suggest_synonyms("happy", service="huggingface", model_name="llama2-7b")
    print("Llama2 Synonyms:", synonyms_llama2)

    synonyms_huggingface = synonym_suggester.suggest_synonyms("happy", service="huggingface", model_name="facebook/nllb-200-distilled-1.3B")
    print("HuggingFace Synonyms:", synonyms_huggingface)

    synonyms_thesaurus = synonym_suggester.suggest_synonyms("happy", service="thesaurus")
    print("Thesaurus Synonyms:", synonyms_thesaurus)

    synonyms_llama2 = synonym_suggester.suggest_synonyms("happy", service="huggingface", model_name="llama2-7b")
    print("Llama2 Synonyms:", synonyms_llama2)

    synonyms_openai = synonym_suggester.suggest_synonyms("happy", service="openai")
    print("OpenAI Synonyms:", synonyms_openai)

except ValueError as e:
    print(e)
