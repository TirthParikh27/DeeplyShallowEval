"""
Vector-based retriever implementation.
"""

from typing import List, Optional
from llama_index.core import (
    VectorStoreIndex,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeWithScore, Node
from llama_index.core.retrievers import BaseRetriever as LlamaBaseRetriever
from src.retrieval.base_retriever import BaseRetriever


class VectorRetriever(BaseRetriever):
    """Vector-based retriever for documents."""

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        similarity_top_k: int = 5,
        persist_path: str = "./storage/vector_index",
    ):
        """
        Initialize the vector-based retriever.

        Args:
            embedding_model: Embedding model to use.
            similarity_top_k: Default number of documents to retrieve.
        """
        self.embedding_model = embedding_model
        self.similarity_top_k = similarity_top_k
        self.persist_path = persist_path
        self.index = None
        self.service_context = None
        self._retriever = None  # Internal llama_index BaseRetriever

    def index_nodes(self, nodes: List[Node]) -> None:
        """
        Index documents for retrieval.

        Args:
            documents: List of documents to index.
        """

        self.index = VectorStoreIndex(nodes=nodes, embed_model=self.embedding_model)

        # Create retriever from index
        self._retriever = self.index.as_retriever(
            similarity_top_k=self.similarity_top_k
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[NodeWithScore]:
        """
        Retrieve documents relevant to the query.

        Args:
            query: Query string.
            top_k: Number of documents to retrieve (overrides default).
            filters: Optional filters to apply to the retrieval.

        Returns:
            List of relevant documents.
        """
        if self._retriever is None:
            raise ValueError("Retriever not initialized. Call index_nodes first.")

        # Create a new retriever with updated parameters if needed
        if top_k is not None and top_k != self.similarity_top_k:
            retriever = self.index.as_retriever(similarity_top_k=top_k)
        else:
            retriever = self._retriever

        # Retrieve nodes directly with the query string
        nodes_with_scores = retriever.retrieve(query)

        return nodes_with_scores

    def get_retriever(self) -> LlamaBaseRetriever:
        """
        Get the internal llama_index retriever.

        Returns:
            The internal LlamaBaseRetriever instance.
        """
        return self._retriever

    def get_index(self) -> VectorStoreIndex:
        """
        Get the internal vector store index.

        Returns:
            The internal VectorStoreIndex instance.
        """
        return self.index
