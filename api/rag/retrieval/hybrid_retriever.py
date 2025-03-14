"""
Hybrid retriever implementation combining vector search and BM25.
"""

from typing import List
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import (
    QueryFusionRetriever,
    BaseRetriever as LlamaBaseRetriever,
)
from llama_index.core.schema import NodeWithScore, Node
from llama_index.core.embeddings import BaseEmbedding
from rag.retrieval.vector_retriever import VectorRetriever
from rag.retrieval.base_retriever import BaseRetriever


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining vector search and BM25 for insurance policy documents."""

    def __init__(
        self,
        embedding_model: BaseEmbedding,
        similarity_top_k: int = 2,
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
