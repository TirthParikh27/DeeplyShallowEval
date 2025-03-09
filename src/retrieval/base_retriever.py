"""
Base retriever interface for insurance policy documents.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
from llama_index.core.schema import NodeWithScore
from llama_index.core.schema import Node
from llama_index.core.retrievers import BaseRetriever as LlamaBaseRetriever


class BaseRetriever(ABC):
    """Base class for document retrievers."""

    @property
    def retriever(self) -> Optional[LlamaBaseRetriever]:
        """
        Get the internal llama_index retriever.

        Returns:
            The internal LlamaBaseRetriever instance or None if not initialized.
        """
        return getattr(self, "_retriever", None)

    @abstractmethod
    def index_nodes(self, nodes: List[Node]) -> None:
        """
        Index documents for retrieval.

        Args:
            documents: List of documents to index.
        """

    @abstractmethod
    def retrieve(self, query: str, first_page, last_page) -> List[NodeWithScore]:
        """
        Retrieve documents relevant to the query.

        Args:
            query: Query string.
            first_page: Starting page number for retrieval.
            last_page: Ending page number for retrieval.

        Returns:
            List of relevant documents with similarity scores.
        """

    @abstractmethod
    def get_retriever(self) -> LlamaBaseRetriever:
        """
        Get the internal llama_index retriever.

        Returns:
            The internal LlamaBaseRetriever instance.
        """
