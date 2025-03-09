from typing import List
from pathlib import Path
from llama_index.core import Document
from llama_parse import LlamaParse


class PDFLoader:
    """PDF document loader using LlamaParse."""

    def __init__(self, api_key: str = None):
        """
        Initialize the PDF loader.

        Args:
            api_key: LlamaParse API key (can also be set via LLAMA_CLOUD_API_KEY env var)
        """
        self.parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            use_vendor_multimodal_model=True,
        )

    async def load_document(self, file_path: str) -> List[Document]:
        """
        Load a single PDF document.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of Document objects
        """

        file_name = Path(file_path).name
        documents = await self.parser.aload_data(file_path)
        for page in documents:
            page.metadata["document_name"] = file_name

        return documents

    async def load_documents(
        self,
        document_paths: List[str],
    ) -> List[Document]:
        """
        Load multiple PDF documents from a directory.

        Args:
            directory_path: Path to directory containing PDF files
            recursive: If True, search recursively in subdirectories

        Returns:
            List of Document objects
        """

        documents = []
        for pdf_file in document_paths:
            new_docs = await self.load_document(pdf_file)
            documents.extend(new_docs)

        return documents
