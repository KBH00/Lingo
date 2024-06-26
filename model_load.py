from exception import ServiceNotFoundException
import model_load as model_load
from typing import Dict, Any
from exception import (
    ModuleNotFoundException,
)
import yaml
import os
import models as models
import importlib.util

def _get_services():
    with open("service.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

_ALL_SERVICES = _get_services()
MODEL_SERVICE_MAPPING_NAME = _ALL_SERVICES["model_services"]

class ModelLoader:

    MODEL_MAPPING: Dict[str, str] = MODEL_SERVICE_MAPPING_NAME

    def __init__(self, model: str):
        self.model = model

    def get_class_from_module(self, module, name):
        module_class = getattr(module, name, None)
        if module_class is None:
            raise ModuleNotFoundException(f"'{name}' class does not exist.")
        return module_class

    def load_class(self, module_file, class_name):
        try:
            module_path = os.path.join("models", f"{module_file}.py")
            
            if not os.path.exists(module_path):
                raise ModuleNotFoundException(f"Module file '{module_file}.py' does not exist.")

            module = importlib.import_module(f"models.{module_file}")
            model_class = self.get_class_from_module(module, class_name)
            
            return model_class

        except ModuleNotFoundException as mnfe:
            print(f"ModuleNotFoundException: {mnfe}")
            raise
        except Exception as e:
            print(f"An error occurred: {e}")
            raise

    def model_return(self):
        if self.model in self.MODEL_MAPPING:
            model_class = self.load_class(self.model, self.MODEL_MAPPING[self.model])
            return model_class()
        else:
            raise ServiceNotFoundException(
                f"{self.model} does not exist in yaml, please add a model or change the name"
            )
