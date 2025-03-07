from typing import List
from copy import deepcopy
from llama_index.core.node_parser import MarkdownElementNodeParser
from llama_index.core import Document
from llama_index.core.schema import Node, TextNode


class MarkdownProcessor:
    """
    A class to process markdown files from llama parse output.
    """

    def __init__(self, parsed_documents: List[Document]):
        """
        Initialize the processor with parsed documents.

        Args:
            parsed_documents: List of Document objects to process
        """
        self.documents = parsed_documents
        self.md_parser = MarkdownElementNodeParser()
        self.page_nodes = []
        self.nodes = []
        self.objects = []

    def get_page_nodes(self, docs, separator="\n---\n"):
        """
        Split each document into page node, by separator.
        """
        nodes = []
        for doc in docs:
            doc_chunks = doc.text.split(separator)
            for doc_chunk in doc_chunks:
                node = TextNode(
                    text=doc_chunk,
                    metadata=deepcopy(doc.metadata),
                )
                nodes.append(node)

        return nodes

    def parse_documents(self) -> List[Node]:
        """
        Parse loaded documents into nodes using the markdown parser.

        Returns:
            List[Node]: A combined list of all nodes and objects
        """
        all_nodes = []
        for doc in self.documents:
            nodes = self.md_parser.get_nodes_from_documents([doc])
            for node in nodes:
                # Add document metadata to each node
                if "document_name" in doc.metadata:
                    node.metadata["document_name"] = doc.metadata["document_name"]

            all_nodes.extend(nodes)

        self.nodes = all_nodes
        self.nodes, self.objects = self.md_parser.get_nodes_and_objects(self.nodes)
        self.page_nodes = self.get_page_nodes(self.documents)

        # Combine nodes and objects into a single list
        combined_nodes = self.nodes + self.objects + self.page_nodes

        return combined_nodes
