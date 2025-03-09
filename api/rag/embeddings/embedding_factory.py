"""
Factory for creating embedding models directly from LlamaIndex.
"""

from typing import Optional, Any

from llama_index.core.embeddings import BaseEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.cohere import CohereEmbedding
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from config import COHERE_API_KEY, OPENAI_API_KEY


class EmbeddingFactory:
    """Factory for creating LlamaIndex embedding models."""

    @staticmethod
    def create_embedding_model(
        provider: str,
        model_name: Optional[str] = None,
        embed_batch_size: int = 10,
        **kwargs: Any,
    ) -> BaseEmbedding:
        """
        Create an embedding model based on the provider.

        Args:
            provider: Embedding provider ("openai" or "cohere").
            model_name: Name of the embedding model (provider-specific).
            embed_batch_size: Batch size for embedding requests.
            kwargs: Additional keyword arguments for the embedding model.

        Returns:
            A LlamaIndex BaseEmbedding instance.

        Raises:
            ValueError: If the provider is not supported.
        """
        provider = provider.lower()

        if provider == "openai":
            default_model = "text-embedding-3-large"
            return OpenAIEmbedding(
                model=model_name or default_model,
                api_key=OPENAI_API_KEY,
                embed_batch_size=embed_batch_size,
                **kwargs,
            )
        elif provider == "cohere":
            default_model = "embed-english-v3.0"
            return CohereEmbedding(
                model_name=model_name or default_model,
                api_key=COHERE_API_KEY,
                embed_batch_size=embed_batch_size,
                **kwargs,
            )
        elif provider == "huggingface":
            default_model = "BAAI/bge-large-en-v1.5"
            return HuggingFaceEmbedding(
                model_name=model_name or default_model,
                embed_batch_size=embed_batch_size,
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
