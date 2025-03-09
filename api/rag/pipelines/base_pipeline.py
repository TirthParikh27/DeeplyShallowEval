"""
Base pipeline interface for RAG systems.
"""

from abc import ABC, abstractmethod
from typing import List
from llama_index.core import Response
from llama_index.core.schema import Document, BaseNode


class BasePipeline(ABC):
    """Base class for RAG pipelines."""

    @abstractmethod
    async def load_pipeline(self, file_paths: List[str]) -> None:
        """
        Load the pipeline by instanciating embedder, retriver and query engine.

        Args:
            file_paths: List of file paths to load.
        """

    @abstractmethod
    def query(self, question: str, **kwargs) -> Response:
        """
        Generate an answer to a question.

        Args:
            question: Question to answer.
            **kwargs: Additional arguments.

        Returns:
            LlamaIndex Response object.
        """

    @abstractmethod
    def get_documents(self) -> List[Document]:
        """
        Get the documents loaded in the pipeline.

        Returns:
            List of Document objects.
        """

    @abstractmethod
    def get_nodes(self) -> List[BaseNode]:
        """
        Get the nodes loaded in the pipeline.

        Returns:
            List of BaseNode objects.
        """
