import os
import pickle
from typing import Any, List, Optional
import hashlib

from config import LLAMA_API_KEY
from src.data.processor import MarkdownProcessor
from src.data.loader import PDFLoader
from src.embeddings.embedding_factory import EmbeddingFactory
from src.generation.llm_factory import LLMFactory
from src.pipelines.base_pipeline import BasePipeline
from src.retrieval.retriever_factory import RetrieverFactory
from llama_index.core.schema import Document, BaseNode
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent
from llama_index.core import Response
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.query_engine import RetrieverQueryEngine


class AgenticRAGPipeline(BasePipeline):
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
        self.query_engine = None
        self.agent = None
        self.tools = None
        self.source_nodes = []

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
            # Step 1 : Load documents
            loader = PDFLoader(api_key=LLAMA_API_KEY)
            self.documents = await loader.load_documents(file_paths)

            # Store processed documents
            os.makedirs(os.path.dirname(persisted_docs_path), exist_ok=True)
            with open(persisted_docs_path, "wb") as f:
                pickle.dump(self.documents, f)
            print(f"Loaded documents and saved to {persisted_docs_path}")

            # Step 2 : Process documents
            self.nodes = await MarkdownProcessor(self.documents).parse_documents()

            # Store processed nodes
            os.makedirs(os.path.dirname(persisted_nodes_path), exist_ok=True)
            with open(persisted_nodes_path, "wb") as f:
                pickle.dump(self.nodes, f)
            print(f"Processed documents and saved nodes to {persisted_nodes_path}")

        # Step 3 : Index documents
        self.retriever.index_nodes(self.nodes)

        # Step 4 : Load generator
        # CHANGE: Remove ResponseMode.CONTEXT_ONLY to allow full use of context
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever.get_retriever(),
            response_mode=ResponseMode.CONTEXT_ONLY,
            llm=self.llm,
        )

        # Step 5 : Create tools
        self.tools = [
            QueryEngineTool.from_defaults(
                query_engine=self.query_engine,
                name="vector_search_tool",
                description="""Useful for querying information from documents.
            Use a detailed plain text question as input to the tool.""",
            )
        ]

        # Step 6 : Create Agent with improved context
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            context="""You are a precise research assistant answering questions from documents.
            For each question:
            1. First, always use the retrieve tool to get relevant information BEFORE attempting to reason about the answer
            2. Only break complex questions into sub-queries if a direct retrieval doesn't yield sufficient information
            3. Heavily rely on the exact information from retrieved content
            4. Synthesize information faithfully without adding any information not present in the retrieved content
            5. If you're unsure about any aspect, retrieve more information rather than making assumptions
            Only use information from the provided tools to answer the question. If no relevant information is found, respond with 'No information found'.
            Avoid making conclusions that aren't directly supported by the retrieved text.""",
            verbose=True,
            max_iterations=8,
        )

    def query(self, question) -> Response:
        agent_response = self.agent.query(question)
        return agent_response

    def get_query_engine(self) -> Any:
        return self.agent

    def get_documents(self) -> List[Document]:
        return self.documents

    def get_nodes(self) -> List[BaseNode]:
        return self.nodes
