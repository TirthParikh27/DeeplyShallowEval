from typing import List, Optional
import hashlib
from config import LLAMA_API_KEY
from src.data.processor import MarkdownProcessor
from src.data.loader import PDFLoader
from src.embeddings.embedding_factory import EmbeddingFactory
from src.generation.llm_factory import LLMFactory
from src.pipelines.base_pipeline import BasePipeline
from src.retrieval.retriever_factory import RetrieverFactory
from llama_index.core import Response
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import Document, BaseNode
import os
import pickle


class StandardRAGPipeline(BasePipeline):
    def __init__(
        self,
        embedder_type: str,
        embedder_model: Optional[str],
        retriever_type: str,
        generator_llm_type: str,
    ):
        self.embedder = EmbeddingFactory.create_embedding_model(
            embedder_type, embedder_model
        )
        self.retriever = RetrieverFactory.create_retriever(
            retriever_type, self.embedder
        )
        self.llm = LLMFactory.create_llm(generator_llm_type)
        self.documents = None
        self.nodes = None
        self.query_engine = None  # After loading the documents

    async def load_pipeline(
        self, file_paths: List[str], persist_path: str = "./storage"
    ) -> None:


        # Create a stable hash from the file paths
        paths_string = ",".join(sorted(file_paths))
        hash_object = hashlib.md5(paths_string.encode())
        stable_hash = hash_object.hexdigest()

        persisted_nodes_path = os.path.join(persist_path, "nodes", stable_hash + ".pkl")
        persisted_docs_path = os.path.join(persist_path, "docs", stable_hash + ".pkl")
        print("Looking for : ", persisted_nodes_path)
        print("Looking for : ", persisted_docs_path)

        # Check if persisted nodes exist
        if os.path.exists(persisted_nodes_path):
            # Load persisted documents
            with open(persisted_docs_path, "rb") as f:
                self.documents = pickle.load(f)
            print(f"Loaded preprocessed documents from {persisted_docs_path}")

            # Load persisted nodes
            with open(persisted_nodes_path, "rb") as f:
                self.nodes = pickle.load(f)
            print(f"Loaded preprocessed nodes from {persisted_nodes_path}")
        else:
            # Step 1: Load documents
            loader = PDFLoader(api_key=LLAMA_API_KEY)
            self.documents = await loader.load_documents(file_paths)
            
            # Store processed documents
            os.makedirs(os.path.dirname(persisted_docs_path), exist_ok=True)
            with open(persisted_docs_path, "wb") as f:
                pickle.dump(self.documents, f)
            print(f"Loaded documents and saved to {persisted_docs_path}")

            # Step 2: Process documents
            self.nodes = await MarkdownProcessor(self.documents).parse_documents()

            # Store processed nodes
            os.makedirs(os.path.dirname(persisted_nodes_path), exist_ok=True)
            with open(persisted_nodes_path, "wb") as f:
                pickle.dump(self.nodes, f)
            print(f"Processed documents and saved nodes to {persisted_nodes_path}")

        # Step 3: Index documents
        self.retriever.index_nodes(self.nodes)

        # Step 4: Load generator
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever.get_retriever(),
            llm=self.llm,
        )

    def query(self, question: str) -> Response:
        return self.query_engine.query(question)

    def get_query_engine(self) -> RetrieverQueryEngine:
        return self.query_engine

    def get_documents(self) -> List[Document]:
        return self.documents

    def get_nodes(self) -> List[BaseNode]:
        return self.nodes
