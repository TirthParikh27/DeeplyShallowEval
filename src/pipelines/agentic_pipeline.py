import os
import pickle
from typing import List, Optional

from config import LLAMA_API_KEY
from src.data.processor import MarkdownProcessor
from src.data.loader import PDFLoader
from src.embeddings.embedding_factory import EmbeddingFactory
from src.generation.llm_factory import LLMFactory
from src.pipelines.base_pipeline import BasePipeline
from src.retrieval.retriever_factory import RetrieverFactory
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent
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
        self.query_engine = None
        self.agent = None
        self.tools = None

    def load_pipeline(
        self, file_paths: List[str], persist_path: str = "./storage"
    ) -> None:

        persisted_files_path = os.path.join(
            persist_path, str(hash(file_paths.sort())) + ".pkl"
        )

        # Check if persisted nodes exist
        if os.path.exists(persisted_files_path):
            # Load persisted nodes
            with open(persisted_files_path, "rb") as f:
                parsed_nodes = pickle.load(f)
            print(f"Loaded preprocessed nodes from {persisted_files_path}")
        else:
            # Step 1 : Load documents
            loader = PDFLoader(api_key=LLAMA_API_KEY)
            documents = loader.load_documents(file_paths)

            # Step 2 : Process documents
            parsed_nodes = MarkdownProcessor(documents).parse_documents()

            # Store processed nodes
            with open(persisted_files_path, "wb") as f:
                pickle.dump(parsed_nodes, f)
            print(f"Processed documents and saved nodes to {persisted_files_path}")

        # Step 3 : Index documents
        self.retriever.index_nodes(parsed_nodes)

        # Step 4 : Load generator
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever=self.retriever.get_retriever(),
            llm=self.llm,
        )

        # Step 5 : Define Query Engine as tool
        self.tools = [
            QueryEngineTool.from_defaults(
                query_engine=self.query_engine,
                name="vector_search_tool",
                description="""Useful for querying information from documents. 
                Use a detailed plain text question as input to the tool.""",
            )
        ]

        # Step 6 : Create Agent
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm,
            verbose=True,
        )

    def answer_question(self, question):
        return self.agent.query(question)
