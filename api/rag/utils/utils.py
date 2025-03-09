from typing import List
import os
from llama_index.core.schema import BaseNode


def get_unique_sources(nodes: List[BaseNode]) -> List[str]:
    """
    Get a list of unique source names from a list of documents.

    Args:
        nodes: List of BaseNode objects.

    Returns:
        List of unique source names.
    """
    unique_sources = set()
    for node in nodes:
        source_name = node.metadata.get("document_name")
        source_name = source_name.split(".")[0] if source_name else None
        if source_name:
            unique_sources.add(source_name)
    unique_sources = list(unique_sources)
    unique_sources.sort()
    return unique_sources


def get_pdf_paths(data_path: str) -> List[str]:
    """
    Get a list of file names from a list of file paths.

    Args:
        file_paths: List of file paths.

    Returns:
        List of file names.
    """
    # Find PDF files in the specified directory
    pdf_files = []
    if os.path.exists(data_path):
        for file in os.listdir(data_path):
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(data_path, file))

    if not pdf_files:
        print(f"No PDF files found in {data_path}")
    else:
        print(f"Found {len(pdf_files)} PDF files")

    return pdf_files
