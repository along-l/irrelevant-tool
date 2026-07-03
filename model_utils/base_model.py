import importlib
from abc import ABC

from model_utils.model_config import get_model_module


class BaseModel(ABC):
    _registry = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__module__] = cls

    @classmethod
    def create(cls, model_name: str, **kwargs):
        model_module = get_model_module(model_name)
        if model_module not in cls._registry:
            cls._auto_import(model_module)
        if model_module not in cls._registry:
            raise ValueError(f"Unknown model_module: {model_module}. Available: {list(cls._registry.keys())}")
        return cls._registry[model_module](model_name, **kwargs)

    @classmethod
    def _auto_import(cls, model_module: str):
        try:
            print(f"Importing model module: {model_module}")
            importlib.import_module(model_module)
        except ImportError:
            pass
