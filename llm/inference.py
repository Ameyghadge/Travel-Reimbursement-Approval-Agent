"""Inference engine providing a clean generate(prompt) -> str interface."""

import torch
import structlog
from llm.loader import ModelLoader

logger = structlog.get_logger()


class InferenceEngine:
    """Generates text using the loaded Qwen model.
    Optimized for low-latency CPU inference."""

    def __init__(self, loader: ModelLoader):
        self._loader = loader
        self._model = loader.model
        self._tokenizer = loader.tokenizer
        self._device = loader.device

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response from the given prompt.

        Uses greedy decoding (temperature=0) for speed.
        Returns only the newly generated text.
        """
        messages = [
            {"role": "user", "content": prompt},
        ]

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self._device)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            if temperature <= 0.01:
                # Greedy decoding — fastest
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            else:
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_k=10,
                    pad_token_id=self._tokenizer.eos_token_id,
                )

        generated_ids = outputs[0][input_length:]
        response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        logger.debug(
            "llm_generated",
            prompt_tokens=input_length,
            response_tokens=len(generated_ids),
        )
        return response.strip()
