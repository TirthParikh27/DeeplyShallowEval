"""
Base pipeline interface for RAG systems.
"""

from abc import ABC, abstractmethod
from typing import List
from llama_index.core import Response


class BasePipeline(ABC):
    """Base class for RAG pipelines."""

    @abstractmethod
    def load_pipeline(self, file_paths: List[str]) -> None:
        """
        Load the pipeline by instanciating embedder, retriver and query engine.

        Args:
            file_paths: List of file paths to load.
        """

    @abstractmethod
    def answer_question(self, question: str, **kwargs) -> Response:
        """
        Generate an answer to a question.

        Args:
            question: Question to answer.
            **kwargs: Additional arguments.

        Returns:
            LlamaIndex Response object.
        """
