"""
Factory class for creating different types of retrievers.
"""

from llama_index.core.embeddings import BaseEmbedding
from src.retrieval.base_retriever import BaseRetriever
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever


class RetrieverFactory:
    """Factory class for creating retriever instances."""

    @staticmethod
    def create_retriever(
        retriever_type: str,
        embedding_model: BaseEmbedding,
        similarity_top_k: int = 2,
    ) -> BaseRetriever:
        """
        Create a retriever instance based on the specified type.

        Args:
            retriever_type: Type of retriever to create ("vector", "hybrid").
            embedding_model: Embedding model to use for the retriever.
            similarity_top_k: Default number of documents to retrieve.
            **kwargs: Additional keyword arguments for specific retriever types.

        Returns:
            An instance of the requested retriever type.

        Raises:
            ValueError: If the requested retriever type is not supported.
        """
        retriever_type = retriever_type.lower()

        if retriever_type == "vector":
            return VectorRetriever(
                embedding_model=embedding_model, similarity_top_k=similarity_top_k
            )
        elif retriever_type == "hybrid":
            return HybridRetriever(
                embedding_model=embedding_model, similarity_top_k=similarity_top_k
            )

        else:
            raise ValueError(
                f"Unsupported retriever type: {retriever_type}. "
                f"Supported types are: 'vector', 'hybrid'"
            )
