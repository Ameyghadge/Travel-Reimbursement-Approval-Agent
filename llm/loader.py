"""Singleton model and tokenizer loader for Qwen2.5-3B-Instruct."""



import torch
import structlog
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = structlog.get_logger()


class ModelLoader:
    """Loads the LLM and tokenizer exactly once. Singleton pattern."""

    _instance = None
    _model = None
    _tokenizer = None

    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B-Instruct"):
        self.model_name = model_name
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @classmethod
    def get_instance(cls, model_name: str = "Qwen/Qwen2.5-3B-Instruct") -> "ModelLoader":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls(model_name)
        return cls._instance

    @property
    def device(self) -> torch.device:
        return self._device

    def load(self) -> tuple:
        """Load model and tokenizer. Called once at startup."""
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer

        logger.info("loading_model", model=self.model_name, device=str(self._device))

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            dtype = torch.float16 if self._device.type == "cuda" else torch.float32

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=dtype,
                device_map="auto" if self._device.type == "cuda" else None,
                trust_remote_code=True,
            )

            if self._device.type == "cpu":
                self._model = self._model.to(self._device)

            self._model.eval()
            logger.info("model_loaded", model=self.model_name, device=str(self._device))

        except Exception as e:
            logger.error("model_load_failed", error=str(e))
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}") from e

        return self._model, self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self.load()
        return self._tokenizer
