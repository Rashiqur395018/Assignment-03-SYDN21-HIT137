from abc import ABC, abstractmethod
from utils import logged, timed
from transformers import pipeline
import threading

# Base interface for models:
class ModelInterface(ABC):
    @abstractmethod
    def predict(self, input_data):
        """Run inference on input_data and return structured output."""
        raise NotImplementedError()

    @abstractmethod
    def get_info(self):
        """Return a short description of the model and expected inputs."""
        raise NotImplementedError()

class LoggingMixin:
    def log(self, message):
        # Simple print-based logging; could be extended to write to a file
        print(f"[{self._class.name_}] {message}")

# Text sentiment model:
class TextSentimentModel(ModelInterface, LoggingMixin):
    def _init_(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self._model_name = model_name
        self._pipeline = None
        self._lock = threading.Lock()

    def _ensure_pipeline(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    self.log(f"Loading text pipeline ({self._model_name})...")
                    self._pipeline = pipeline("text-classification", model=self._model_name)

    @timed
    @logged("TextSentimentModel.predict")
    def predict(self, text):
        """Override: run sentiment classification on input text."""
        self._ensure_pipeline()
        results = self._pipeline(text, truncation=True)
        return results

    def get_info(self):
        return (
            f"Model name: {self._model_name}\n"
            "Task: Text classification (sentiment)\n"
            "Input: a short text string.\n"
            "Output: list of label + score dicts from Hugging Face pipeline."
        )
