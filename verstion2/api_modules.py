import requests
from abc import ABC, abstractmethod
import os

API_SYNONYM_SERVICE_MAPPING_NAME = {
    "thesaurus": "ThesaurusAPI",
    # Add other API services here
}

class AbstractAPI(ABC):
    @abstractmethod
    def suggest_synonyms(self, word: str) -> dict:
        pass

class ThesaurusAPI(AbstractAPI):
    def __init__(self):
        self.api_key = os.getenv("THESAURUS_API_KEY")
        self.base_url = "https://www.dictionaryapi.com/api/v3/references/thesaurus/json"

    def suggest_synonyms(self, word: str) -> dict:
        url = f"{self.base_url}/{word}?key={self.api_key}"
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"API request failed with status code {response.status_code}")
        
        data = response.json()
        if not data or not isinstance(data, list) or not isinstance(data[0], dict):
            return {"synonyms": []}

        synonyms = data[0].get('meta', {}).get('syns', [[]])[0]
        return {"synonyms": synonyms}
