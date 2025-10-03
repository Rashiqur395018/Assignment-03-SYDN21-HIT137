from abc import ABC, abstractmethod
from utils import logged, timed
from transformers import pipeline
import threading

class ModelInterface(ABC):
    @abstractmethod
    def predict(self, input_data):
        raise NotImplementedError()

    @abstractmethod
    def get_info(self):
        raise NotImplementedError()

class LoggingMixin:
    def log(self, message):
        print(f"[{self.__class__.__name__}] {message}")

class TextSentimentModel(ModelInterface, LoggingMixin):
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
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

class ImageClassificationModel(ModelInterface, LoggingMixin):
    def __init__(self, model_name="google/vit-base-patch16-224"):
        self._model_name = model_name
        self._pipeline = None
        self._lock = threading.Lock()

    def _ensure_pipeline(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    self.log(f"Loading image pipeline ({self._model_name})...")
                    self._pipeline = pipeline("image-classification", model=self._model_name)

    @timed
    @logged("ImageClassificationModel.predict")
    def predict(self, image_path):
        """Override: run image classification on an image path."""
        self._ensure_pipeline()
        results = self._pipeline(image_path, top_k=5)
        return results

    def get_info(self):
        return (
            f"Model name: {self._model_name}\n"
            "Task: Image classification\n"
            "Input: image file path (png, jpg, bmp)\n"
            "Output: list of labels with scores (top-k)."
        )
