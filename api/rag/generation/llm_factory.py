from typing import Optional
from llama_index.llms.openai import OpenAI
from llama_index.llms.mistralai import MistralAI
from llama_index.core.llms import LLM
from llama_index.llms.anthropic import Anthropic


class LLMFactory:
    """Factory class for creating LlamaIndex LLM instances."""

    @staticmethod
    def create_llm(
        llm_type: str,
        model: Optional[str] = None,
        temperature: float = 0.1,
    ) -> LLM:
        """
        Create and return a LlamaIndex LLM instance based on the specified type.

        Args:
            llm_type (str): The type of LLM to create ('openai', 'mistral', etc.)
            model (str, optional): The specific model to use
            temperature (float): The sampling temperature
            api_key (str, optional): API key for the service

        Returns:
            LLM: A LlamaIndex LLM instance

        Raises:
            ValueError: If the specified LLM type is not supported
        """
        llm_type = llm_type.lower()

        if llm_type == "openai":
            default_model = model or "gpt-4o"
            return OpenAI(
                model=default_model,
                temperature=temperature,
            )

        elif llm_type == "mistral":
            default_model = model or "mistral-small-latest"
            return MistralAI(
                model=default_model,
                temperature=temperature,
            )
        elif llm_type == "anthropic":
            default_model = model or "claude-3-5-sonnet-latest"

            return Anthropic(
                model=default_model,
                temperature=temperature,
            )

        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")
