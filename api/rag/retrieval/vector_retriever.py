"""
Vector-based retriever implementation.
"""

from typing import List
from llama_index.core import (
    VectorStoreIndex,
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.schema import NodeWithScore, Node
from llama_index.core.retrievers import BaseRetriever as LlamaBaseRetriever
from rag.retrieval.base_retriever import BaseRetriever


class VectorRetriever(BaseRetriever):
    """Vector-based retriever for documents."""

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        similarity_top_k: int = 2,
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
        first_page,
        last_page,
    ) -> List[NodeWithScore]:
        """
        Retrieve documents relevant to the query.

        Args:
            query: Query string.
            first_page: Starting page number for retrieval.
            last_page: Ending page number for retrieval.

        Returns:
            List of relevant documents with similarity scores.
        """
        if self._retriever is None:
            raise ValueError("Retriever not initialized. Call index_nodes first.")

        # Retrieve nodes directly with the query string
        nodes_with_scores = self._retriever.retrieve(query)

        # Apply pagination
        start_idx = first_page
        end_idx = min(last_page, len(nodes_with_scores))
        return nodes_with_scores[start_idx:end_idx]

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
