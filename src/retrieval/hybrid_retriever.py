"""
Hybrid retriever implementation combining vector search and BM25.
"""

from typing import List, Optional
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import (
    QueryFusionRetriever,
    BaseRetriever as LlamaBaseRetriever,
)
from llama_index.core.schema import NodeWithScore, Node
from llama_index.core.embeddings import BaseEmbedding
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.base_retriever import BaseRetriever


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining vector search and BM25 for insurance policy documents."""

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        similarity_top_k: int = 5,
        persist_path: str = "./storage/bm25_retriever",
    ):
        """
        Initialize the hybrid retriever.

        Args:
            embedding_model: Embedding model to use.
            similarity_top_k: Default number of documents to retrieve.
        """
        self.embedding_model = embedding_model
        self.similarity_top_k = similarity_top_k
        self.vector_weight = 0.6
        self.bm25_weight = 0.4
        self.nodes = None
        self.vector_retriever = VectorRetriever(embedding_model, similarity_top_k)
        self.bm25_retriever = None
        self.persist_path = persist_path
        self._retriever = None

    def index_nodes(self, nodes: List[Node]) -> None:
        """
        Index documents for retrieval.

        Args:
            documents: List of documents to index.
        """
        # Store documents for BM25 indexing
        self.nodes = nodes

        # Index documents in vector retriever
        self.vector_retriever.index_nodes(self.nodes)

        self.bm25_retriever = BM25Retriever.from_defaults(
            index=self.vector_retriever.get_index(),
            similarity_top_k=self.similarity_top_k,
        )

        # Create fusion retriever
        self._retriever = QueryFusionRetriever(
            retrievers=[self.vector_retriever.get_retriever(), self.bm25_retriever],
            retriever_weights=[self.vector_weight, self.bm25_weight],
            similarity_top_k=self.similarity_top_k,
            num_queries=1,
            mode="relative_score",
            use_async=False,
            verbose=True,
        )

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[NodeWithScore]:
        """
        Retrieve documents relevant to the query using hybrid approach.

        Args:
            query: Query string.
            top_k: Number of documents to retrieve (overrides default).
            filters: Optional filters to apply to the retrieval.

        Returns:
            List of relevant documents.
        """
        if self._retriever is None:
            raise ValueError("Index not initialized. Call index_nodes first.")

        # Set final top_k parameter
        if top_k is not None:
            self._retriever.similarity_top_k = top_k

        # Use the fusion retriever to get combined results
        nodes_with_scores = self._retriever.retrieve(query)

        return nodes_with_scores

    def get_retriever(self) -> LlamaBaseRetriever:
        """
        Get the internal llama_index retriever.

        Returns:
            The internal LlamaBaseRetriever instance.
        """
        return self._retriever
